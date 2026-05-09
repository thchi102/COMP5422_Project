"""
run_vlm_grounder.py
───────────────────
Wrapper that runs the VLM-Grounder baseline pipeline on our dev-mini queries
and saves predictions in schema v1 reconstruction format.

Pipeline stages (each skipped automatically if output already exists):
  Stage 0a  Extract posed images from .sens          (one-time per scene)
  Stage 0b  Generate scene info pkl                  (one-time)
  Stage 2   PATS exhaustive matching                 (one-time per scene)
  Stage 3   Query analysis  (GPT / local LLM)        (per run)
  Stage 4   Instance detection  (YOLO-World)         (per run)
  Stage 5   View pre-selection                       (per run)
  Stage 6   Visual grounding   (GPT / local LLM)     (per run)

Output: outputs/<run_id>/reconstruction/<scene_id>/<sample_id>.json
        one file per query, schema v1 reconstruction package.

─────────────────────────────────────────────────
CONFIGURE THESE PATHS BEFORE RUNNING
─────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import hashlib
import pickle
import platform
from pathlib import Path

# ── Path configuration ────────────────────────────────────────────────────────
# Adjust these to match the server layout.
# VLM_GROUNDER_REPO  : where VLM-Grounder is cloned  (gitignored, clone manually)
# SCANNET_DATA_ROOT  : our ScanNet data root — VLM-Grounder will be pointed here
# ─────────────────────────────────────────────────────────────────────────────

def find_project_root(marker: str = "data/schema_v1.json") -> Path:
    for p in [Path.cwd()] + list(Path.cwd().parents)[:6]:
        if (p / marker).exists():
            return p.resolve()
    raise RuntimeError("Cannot find project root. Run from inside the project directory.")


PROJECT          = find_project_root()
VLM_GROUNDER_REPO = PROJECT / "scripts" / "vlm-grounder-repo"   # git-cloned externally
SCANNET_DATA_ROOT  = PROJECT / "data" / "ScanNet"               # our ScanNet root
REFERIT3D_ROOT     = PROJECT / "data" / "ReferIt3D"
SCAN_DOWNLOAD_SCRIPT = SCANNET_DATA_ROOT / "scannet_download.py"

# VLM-Grounder's expected data layout — symlink or configure below
VG_SCANNET_ROOT    = VLM_GROUNDER_REPO / "data" / "scannet"
VG_SCANS_DIR       = VG_SCANNET_ROOT / "scans"
VG_POSED_DIR       = VG_SCANNET_ROOT / "posed_images"
VG_INSTANCE_DATA   = VG_SCANNET_ROOT / "scannet_instance_data"
VG_MATCH_DATA      = VG_SCANNET_ROOT / "scannet_match_data"

# ── LLM backend — quick-switch config ────────────────────────────────────────
#
# PRIMARY   → vllm   : google/gemma-4-E4B-it served locally, no API cost
#             Switch: DEFAULT_LLM_BACKEND = "vllm"  (already set)
#
# FALLBACK  → gemini : Gemini 2.0 Flash cloud API, free tier
#             Switch: DEFAULT_LLM_BACKEND = "gemini"  +  export GEMINI_API_KEY=...
#
# PAPER     → openai : GPT-4o (matches paper results, costs money)
#             Switch: DEFAULT_LLM_BACKEND = "openai"  +  export OPENAI_API_KEY=...
#
# To change model without changing backend, edit the constant below
# (or pass --llm-model / --vllm-model at the CLI).
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_LLM_BACKEND = "vllm"
DEFAULT_LLM_MODEL   = "gpt-4o-2024-05-13"      # used when backend=openai
OLLAMA_MODEL        = "qwen2-vl:7b"             # used when backend=ollama
GEMINI_MODEL        = "gemini-2.0-flash"        # used when backend=gemini
VLLM_MODEL          = "google/gemma-4-E4B-it"  # used when backend=vllm  ← primary
VLLM_PORT           = 8000                      # vLLM server port
VLLM_QUANTIZATION   = None                      # None=BF16 full; "bitsandbytes"=INT4; "awq"=AWQ
VLLM_CONDA_ENV      = "vllm_server"             # separate conda env (needs CUDA 11.8+)
VLLM_GPU_MEM_UTIL   = 0.90                      # fraction of GPU VRAM for vLLM (0.40 ≈ 9.8 GB on 24 GB)
VLLM_MAX_MODEL_LEN  = None                      # None = let vLLM decide; set int (e.g. 4096) to cap KV cache needs
VLLM_MAX_NUM_SEQS   = 1                         # lower concurrency for shared GPU stability
VLLM_MAX_BATCHED_TOKENS = 512                   # conservative prefill budget for shared GPU stability
VLLM_STARTUP_RETRIES = 2                        # additional retries after initial startup attempt


# ── Helpers ───────────────────────────────────────────────────────────────────

SEP = "=" * 60
STOP_SENTINEL = PROJECT / "outputs" / "STOP"

def log(msg: str = "") -> None:
    print(msg, flush=True)

def log_stage(name: str) -> None:
    log(); log(SEP); log(f"  {name}"); log(SEP)

def check_stop_sentinel() -> None:
    """Raise SystemExit if outputs/STOP exists (graceful kill switch).

    Usage on server:
        touch /path/to/project/outputs/STOP    # stop after current stage
        rm    /path/to/project/outputs/STOP    # clear before next run
    The try/finally in main() ensures vLLM server is terminated on any exit.
    """
    if STOP_SENTINEL.exists():
        log()
        log(f"  [STOP] Sentinel file detected: {STOP_SENTINEL}")
        log(f"  [STOP] Exiting cleanly between stages. vLLM server will be shut down.")
        log(f"  [STOP] Remove the file before next run: rm {STOP_SENTINEL}")
        raise SystemExit(0)

def run_cmd(cmd: list[str], cwd: Path, env: dict | None = None) -> int:
    """Run a subprocess, stream output, return exit code."""
    full_env = {**os.environ, **(env or {})}
    repo_py_path = str(VLM_GROUNDER_REPO)
    existing_py_path = full_env.get("PYTHONPATH", "")
    if existing_py_path:
        py_path_parts = existing_py_path.split(os.pathsep)
        if repo_py_path not in py_path_parts:
            full_env["PYTHONPATH"] = f"{repo_py_path}{os.pathsep}{existing_py_path}"
    else:
        full_env["PYTHONPATH"] = repo_py_path
    log(f"  cmd: {' '.join(str(c) for c in cmd)}")
    log(f"  cwd: {cwd}")
    log()
    result = subprocess.run([str(c) for c in cmd], cwd=str(cwd), env=full_env)
    return result.returncode

def skip_if_exists(path: Path, label: str) -> bool:
    if path.exists():
        log(f"  [SKIP] {label} - already exists: {path}")
        return True
    return False


def _link_dir(link_path: Path, target_path: Path) -> None:
    """
    Create a directory link. Prefer a real symlink; on Windows fall back to a
    junction when symlink privileges are unavailable.
    """
    if link_path.exists() or link_path.is_symlink():
        return
    try:
        link_path.symlink_to(target_path)
        return
    except OSError:
        pass

    if os.name == "nt":
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link_path), str(target_path)],
            capture_output=True,
            text=True,
            errors="ignore",
        )
        if result.returncode == 0 and (link_path.exists() or link_path.is_symlink()):
            return
        raise OSError(
            f"Failed to create junction {link_path} -> {target_path}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    raise OSError(f"Failed to create directory link {link_path} -> {target_path}")


def _stable_scene_tag(scenes: list[str]) -> str:
    joined = ",".join(sorted(scenes))
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:10]


def _manifest_has_queries_for_scenes(manifest_path: Path, scenes: list[str] | None) -> bool:
    if not manifest_path.exists():
        return False
    if not scenes:
        return True
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    queries = manifest.get("queries", [])
    scene_set = set(scenes)
    return any(q.get("scene_id") in scene_set for q in queries)


def ensure_manifest_for_run(manifest_path: Path, scenes: list[str] | None) -> Path:
    """
    Keep the existing manifest path if it already covers the requested scenes.
    Otherwise, auto-build a manifest from ReferIt3D so the notebook can still
    drive new scenes through the same wrapper path.
    """
    if _manifest_has_queries_for_scenes(manifest_path, scenes):
        return manifest_path

    if not scenes:
        raise FileNotFoundError(
            f"Manifest not usable and no scene subset provided: {manifest_path}"
        )

    nr3d_csv = REFERIT3D_ROOT / "nr3d.csv"
    sr3d_csv = REFERIT3D_ROOT / "sr3d+.csv"
    if not nr3d_csv.exists():
        raise FileNotFoundError(
            f"Requested scenes are not in {manifest_path}, and ReferIt3D source is missing: {nr3d_csv}"
        )

    out_dir = PROJECT / "outputs" / "_autogen_manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    auto_manifest = out_dir / f"referit3d_{_stable_scene_tag(scenes)}.json"

    if not auto_manifest.exists():
        log_stage("Manifest prep - building manifest from ReferIt3D")
        rc = run_cmd(
            [
                sys.executable,
                str(PROJECT / "scripts" / "build_referit3d_manifest.py"),
                "--out", str(auto_manifest),
                "--scenes", *scenes,
                "--nr3d", str(nr3d_csv),
                "--sr3d", str(sr3d_csv),
            ],
            cwd=PROJECT,
        )
        if rc != 0:
            raise RuntimeError(f"Failed to auto-build ReferIt3D manifest (exit {rc})")
    else:
        log(f"  [OK] Reusing autogenerated manifest: {auto_manifest}")

    if not _manifest_has_queries_for_scenes(auto_manifest, scenes):
        raise RuntimeError(
            f"Autogenerated manifest has no queries for requested scenes: {scenes}"
        )
    return auto_manifest


def ensure_scannet_scene_assets(scenes: list[str] | None) -> None:
    """
    Download the canonical ScanNet scene assets needed by Stage 0a/0b for the
    requested scenes. This keeps us on VLM-Grounder's standard data path.
    """
    if not scenes:
        log("  [INFO] Auto-download skipped because no explicit scene subset was provided.")
        return
    if not SCAN_DOWNLOAD_SCRIPT.exists():
        raise FileNotFoundError(f"ScanNet downloader not found: {SCAN_DOWNLOAD_SCRIPT}")

    required_types = [
        ".sens",
        ".txt",
        ".aggregation.json",
        "_vh_clean_2.0.010000.segs.json",
        "_vh_clean_2.ply",
        "_vh_clean_2.labels.ply",
    ]
    scans_root = SCANNET_DATA_ROOT / "scans"

    missing = []
    for sid in scenes:
        scene_dir = scans_root / sid
        for suffix in required_types:
            if not (scene_dir / f"{sid}{suffix}").exists():
                missing.append((sid, suffix))

    if not missing:
        log("  [OK] Required ScanNet scene assets already present for requested scenes.")
        return

    log_stage("Stage 0 prep - downloading missing ScanNet scene assets")
    missing_by_scene: dict[str, list[str]] = {}
    for sid, suffix in missing:
        missing_by_scene.setdefault(sid, []).append(suffix)
    for sid in scenes:
        if sid in missing_by_scene:
            log(f"  [MISSING] {sid}: {sorted(missing_by_scene[sid])}")

    scene_list_file = PROJECT / "outputs" / "_autogen_manifests" / f"selected_scenes_{_stable_scene_tag(scenes)}.txt"
    scene_list_file.parent.mkdir(parents=True, exist_ok=True)
    scene_list_file.write_text("\n".join(scenes) + "\n", encoding="utf-8")

    rc = run_cmd(
        [
            sys.executable,
            str(SCAN_DOWNLOAD_SCRIPT),
            "-o", str(SCANNET_DATA_ROOT),
            "--id_file", str(scene_list_file),
            "--types", *required_types,
            "--skip_existing",
            "--yes",
            "--keep_sens",
        ],
        cwd=PROJECT,
    )
    if rc != 0:
        raise RuntimeError(f"ScanNet asset download failed (exit {rc})")

    still_missing = []
    for sid in scenes:
        scene_dir = scans_root / sid
        for suffix in required_types:
            if not (scene_dir / f"{sid}{suffix}").exists():
                still_missing.append(f"{sid}{suffix}")
    if still_missing:
        raise RuntimeError(
            "Missing ScanNet assets after download:\n  " + "\n  ".join(still_missing)
        )
    log("  [DONE] Required ScanNet scene assets are present.")


def _pkl_covers_scenes(pkl_path: Path, scenes: list[str] | None, require_images_info: bool = False) -> bool:
    if not pkl_path.exists():
        return False
    if not scenes:
        return True
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    for sid in scenes:
        if sid not in data:
            return False
        if require_images_info:
            scene_info = data.get(sid, {})
            if not isinstance(scene_info, dict):
                return False
            if "images_info" not in scene_info or "num_posed_images" not in scene_info:
                return False
    return True


def _run_capture(cmd: list[str], cwd: Path | None = None) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            [str(c) for c in cmd],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            errors="ignore",
            timeout=60,
        )
        output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        return proc.returncode, output.strip()
    except Exception as exc:
        return 1, f"{type(exc).__name__}: {exc}"


def write_runtime_snapshot(out_dir: Path, scenes: list[str] | None, manifest_path: Path) -> None:
    """
    Persist a concise environment snapshot inside the run folder so SSH-side
    failures can be debugged from downloaded artifacts alone.
    """
    snapshot_dir = out_dir / "runtime"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    env_keys = [
        "CONDA_DEFAULT_ENV",
        "CONDA_PREFIX",
        "CUDA_VISIBLE_DEVICES",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "DEEPDATASPACE_API_KEY",
        "PYTHONPATH",
    ]
    env_summary = {}
    for key in env_keys:
        value = os.environ.get(key)
        if value is None:
            env_summary[key] = None
        elif "KEY" in key:
            env_summary[key] = f"set(len={len(value)})" if value else "empty"
        else:
            env_summary[key] = value

    git_rc, git_head = _run_capture(["git", "rev-parse", "HEAD"], cwd=PROJECT)
    git_status_rc, git_status = _run_capture(["git", "status", "--short"], cwd=PROJECT)
    python_rc, python_version = _run_capture([sys.executable, "--version"])
    pip_rc, pip_freeze = _run_capture([sys.executable, "-m", "pip", "freeze"])
    nvidia_rc, nvidia_smi = _run_capture(["nvidia-smi"])
    gpu_query_rc, gpu_query = _run_capture(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,driver_version", "--format=csv,noheader"]
    )

    snapshot = {
        "project": str(PROJECT),
        "cwd": str(Path.cwd()),
        "out_dir": str(out_dir),
        "manifest": str(manifest_path),
        "scenes": scenes or "all",
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "python_executable": sys.executable,
        },
        "env": env_summary,
        "commands": {
            "python_version": {"returncode": python_rc, "output": python_version},
            "git_head": {"returncode": git_rc, "output": git_head},
            "git_status": {"returncode": git_status_rc, "output": git_status},
            "nvidia_smi": {"returncode": nvidia_rc, "output": nvidia_smi},
            "nvidia_gpu_query": {"returncode": gpu_query_rc, "output": gpu_query},
        },
    }
    (snapshot_dir / "runtime_snapshot.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    (snapshot_dir / "pip_freeze.txt").write_text(pip_freeze + ("\n" if pip_freeze else ""), encoding="utf-8")


# ── Sanity-check helpers ──────────────────────────────────────────────────────

def _assert_file(path: Path, stage: str, min_bytes: int = 100) -> None:
    """Raise with clear message if output file is missing or suspiciously small."""
    if not path.exists():
        raise RuntimeError(
            f"[{stage}] SANITY FAIL - expected output not found:\n"
            f"         {path}\n"
            f"         Check stage logs above for errors."
        )
    size = path.stat().st_size
    if size < min_bytes:
        raise RuntimeError(
            f"[{stage}] SANITY FAIL - output file too small ({size} bytes):\n"
            f"         {path}"
        )
    log(f"  [SANITY OK] {path.name}  ({size:,} bytes)")


def _assert_csv_rows(
    path: Path, stage: str,
    min_rows: int = 1,
    required_cols: "list[str] | None" = None,
) -> None:
    """Raise if CSV is missing, has too few rows, or is missing required columns."""
    import csv as _csv
    _assert_file(path, stage)
    with open(path, newline="", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        header = list(reader.fieldnames or [])
        rows = list(reader)
    if len(rows) < min_rows:
        raise RuntimeError(
            f"[{stage}] SANITY FAIL - CSV has {len(rows)} rows (expected >={min_rows}):\n"
            f"         {path}"
        )
    if required_cols:
        missing = [c for c in required_cols if c not in header]
        if missing:
            raise RuntimeError(
                f"[{stage}] SANITY FAIL - CSV missing columns {missing}\n"
                f"         Found: {header}\n"
                f"         Path:  {path}"
            )
    col_preview = ", ".join(header[:6]) + ("..." if len(header) > 6 else "")
    log(f"  [SANITY OK] {path.name}  ({len(rows)} rows | cols: {col_preview})")


def _assert_json_list(
    path: Path, stage: str,
    min_items: int = 1,
    required_keys: "list[str] | None" = None,
) -> None:
    """Raise if JSON file is missing or has too few items; warn on missing keys."""
    _assert_file(path, stage)
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and isinstance(data.get("results"), list):
        items = data["results"]
    else:
        items = [v for v in data.values() if isinstance(v, dict)] if isinstance(data, dict) else []
    if len(items) < min_items:
        raise RuntimeError(
            f"[{stage}] SANITY FAIL - JSON has {len(items)} items (expected >={min_items}):\n"
            f"         {path}"
        )
    if required_keys and items:
        first = items[0]
        # gpt_pred_bbox may be at top level OR inside eval_result
        eval_result = first.get("eval_result", {}) or {}
        for k in required_keys:
            if k not in first and k not in eval_result:
                log(f"  [SANITY WARN] [{stage}] Key '{k}' not found in first item (may be nested).")
                log(f"               First item keys: {list(first.keys())}")
    log(f"  [SANITY OK] {path.name}  ({len(items)} items)")


def patch_gdino_api_key() -> None:
    """
    VLM-Grounder hardcodes api_key = "your_api_key" in my_gdino.py.
    Patch it at runtime from DEEPDATASPACE_API_KEY env var.
    """
    api_key = os.environ.get("DEEPDATASPACE_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "DEEPDATASPACE_API_KEY is not set. "
            "Add to ~/.bashrc: export DEEPDATASPACE_API_KEY=<your_token>"
        )
    gdino_file = VLM_GROUNDER_REPO / "vlm_grounder" / "utils" / "my_gdino.py"
    if not gdino_file.exists():
        raise FileNotFoundError(f"my_gdino.py not found: {gdino_file}")
    text = gdino_file.read_text()
    patched = text.replace('api_key = "your_api_key"', f'api_key = "{api_key}"')
    if patched == text and f'api_key = "{api_key}"' not in text:
        log(f"  [WARN] Could not find placeholder in {gdino_file} - may already be patched")
    else:
        gdino_file.write_text(patched)
        log(f"  [OK] Patched DEEPDATASPACE_API_KEY into {gdino_file}")


def patch_openai_setup() -> None:
    """
    my_openai.py's setup_openai() hardcodes api_key = "your_api_key" and
    overwrites OPENAI_API_KEY in the environment, breaking any key we pass in.
    It also crashes if HTTP_PROXY / HTTPS_PROXY are not set.

    Patches applied:
      1. Read API key from OPENAI_API_KEY env var instead of hardcoding.
      2. Do not overwrite OPENAI_API_KEY (remove the os.environ assignment).
      3. Use .get() for HTTP_PROXY / HTTPS_PROXY to avoid KeyError.
    """
    openai_file = VLM_GROUNDER_REPO / "vlm_grounder" / "utils" / "my_openai.py"
    if not openai_file.exists():
        raise FileNotFoundError(f"my_openai.py not found: {openai_file}")
    text = openai_file.read_text()

    # Fix 1 + 2: replace hardcoded key + env overwrite with env read
    text = text.replace(
        'api_key = "your_api_key"\n\n    os.environ["OPENAI_API_KEY"] = api_key',
        'api_key = os.environ.get("OPENAI_API_KEY", "")',
    )

    # Fix 3: HTTP_PROXY / HTTPS_PROXY KeyError
    text = text.replace(
        "f\"[OPENAI] http_proxy: {os.environ['HTTP_PROXY']}. https_proxy: {os.environ['HTTPS_PROXY']}\"",
        "f\"[OPENAI] http_proxy: {os.environ.get('HTTP_PROXY', 'not set')}. "
        "https_proxy: {os.environ.get('HTTPS_PROXY', 'not set')}\"",
    )

    openai_file.write_text(text)
    log(f"  [OK] Patched setup_openai() in {openai_file}")


# ── Data layout setup ─────────────────────────────────────────────────────────

def patch_visual_grounder_json_retry() -> None:
    """
    Keep VLM-Grounder's Stage 6 loop alive when a local OpenAI-compatible VLM
    returns malformed JSON. This does not change the grounding algorithm; it
    treats invalid JSON the same way upstream already treats API failures:
    retry the current substep and eventually mark that query failed.
    """
    vg_file = VLM_GROUNDER_REPO / "vlm_grounder" / "grounder" / "visual_grouder.py"
    if not vg_file.exists():
        raise FileNotFoundError(f"visual_grouder.py not found: {vg_file}")

    text = vg_file.read_text(encoding="utf-8")
    original = text

    bbox_old = (
        '            cost += gpt_response["cost"]\n'
        '            gpt_content_json = json.loads(gpt_response["content"])\n'
        '            gpt_message = {\n'
    )
    bbox_new = (
        '            cost += gpt_response["cost"]\n'
        '            try:\n'
        '                gpt_content_json = json.loads(gpt_response["content"])\n'
        '            except Exception as e:\n'
        '                print(\n'
        '                    f"\\t[GPTSelectBBoxID] {scene_id}: [{query[:20]}] Invalid JSON response. Error: {e}. Retrying..."\n'
        '                )\n'
        '                retry += 1\n'
        '                continue\n'
        '            gpt_message = {\n'
    )
    if bbox_old in text:
        text = text.replace(bbox_old, bbox_new, 1)

    bbox_int_old = (
        '            bbox_index = int(gpt_content_json.get("object_id", -1))\n'
        "            if bbox_index < 0 or bbox_index >= num_candidate_bboxes:\n"
    )
    bbox_int_new = (
        '            try:\n'
        '                bbox_index = int(gpt_content_json.get("object_id", -1))\n'
        '            except Exception:\n'
        '                bbox_index = -1\n'
        "            if bbox_index < 0 or bbox_index >= num_candidate_bboxes:\n"
    )
    if bbox_int_old in text:
        text = text.replace(bbox_int_old, bbox_int_new, 1)

    image_old = (
        '            cost += gpt_response["cost"]\n'
        '            gpt_content = gpt_response["content"]\n'
        "            gpt_content_json = json.loads(gpt_content)\n"
        '            pred_image_id = gpt_content_json.get("target_image_id", -1)\n'
    )
    image_new = (
        '            cost += gpt_response["cost"]\n'
        '            gpt_content = gpt_response["content"]\n'
        "            try:\n"
        "                gpt_content_json = json.loads(gpt_content)\n"
        "            except Exception as e:\n"
        "                print(\n"
        "                    f\"[VLMPredImageID] {scene_id}: [{query[:20]}] Invalid JSON response. Error: {e}. Retrying...\"\n"
        "                )\n"
        "                retry += 1\n"
        "                continue\n"
        '            pred_image_id = gpt_content_json.get("target_image_id", -1)\n'
    )
    if image_old in text:
        text = text.replace(image_old, image_new, 1)

    # Older notebook hotfixes wrapped json.loads() but still re-raised
    # malformed responses, which aborts the whole Stage 6 task loop. Convert
    # those legacy re-raises into normal retry handling.
    legacy_image_raise = '                    raise ValueError(f"Cannot parse JSON object from response: {gpt_content}")\n'
    legacy_image_retry = (
        '                    print(\n'
        '                        f"[VLMPredImageID] {scene_id}: [{query[:20]}] Invalid JSON response. Error: Cannot parse JSON object from response: {gpt_content}. Retrying..."\n'
        '                    )\n'
        '                    retry += 1\n'
        '                    continue\n'
    )
    if legacy_image_raise in text:
        text = text.replace(legacy_image_raise, legacy_image_retry)

    legacy_bbox_raise = '                    raise ValueError(f"Cannot parse JSON object from response: {gpt_response[\'content\']}")\n'
    legacy_bbox_retry = (
        '                    print(\n'
        '                        f"\\t[GPTSelectBBoxID] {scene_id}: [{query[:20]}] Invalid JSON response. Error: Cannot parse JSON object from response: {gpt_response[\'content\']}. Retrying..."\n'
        '                    )\n'
        '                    retry += 1\n'
        '                    continue\n'
    )
    if legacy_bbox_raise in text:
        text = text.replace(legacy_bbox_raise, legacy_bbox_retry)

    if text != original:
        compile(text, str(vg_file), "exec")
        vg_file.write_text(text, encoding="utf-8")
        log(f"  [OK] Patched visual_grouder.py JSON retry handling in {vg_file}")
    else:
        log("  [OK] visual_grouder.py JSON retry handling already present")


def ensure_vg_data_layout() -> None:
    """
    VLM-Grounder expects data at vlm-grounder-repo/data/scannet/scans/.
    We create symlinks pointing to our actual ScanNet data directory.
    Only creates symlinks that don't already exist.
    """
    log_stage("Data layout - symlinking our ScanNet data into VLM-Grounder")

    VG_SCANNET_ROOT.mkdir(parents=True, exist_ok=True)
    VG_MATCH_DATA.mkdir(parents=True, exist_ok=True)
    VG_INSTANCE_DATA.mkdir(parents=True, exist_ok=True)
    canonical_instance_root = SCANNET_DATA_ROOT / "scannet_instance_data"
    canonical_instance_root.mkdir(parents=True, exist_ok=True)

    # scans/ symlink
    our_scans = SCANNET_DATA_ROOT / "scans"
    if not VG_SCANS_DIR.exists():
        _link_dir(VG_SCANS_DIR, our_scans)
        log(f"  [OK] Symlinked scans -> {our_scans}")
    else:
        log(f"  [SKIP] scans symlink already exists")

    # posed_images/ symlink (may not exist yet — Stage 0a creates it)
    our_posed = SCANNET_DATA_ROOT / "posed_images"
    if our_posed.exists() and not VG_POSED_DIR.exists():
        _link_dir(VG_POSED_DIR, our_posed)
        log(f"  [OK] Symlinked posed_images -> {our_posed}")
    elif not our_posed.exists():
        log(f"  [INFO] posed_images not yet extracted - Stage 0a will create it")

    # scene_info canonical copy under data/ScanNet/, mirrored into repo-local path when present
    canonical_scene_info = canonical_instance_root / "scenes_train_val_info_w_images.pkl"
    vg_scene_info = VG_INSTANCE_DATA / "scenes_train_val_info_w_images.pkl"
    if canonical_scene_info.exists() and not vg_scene_info.exists():
        shutil.copy2(canonical_scene_info, vg_scene_info)
        log(f"  [OK] Copied canonical scene info pkl -> {vg_scene_info}")


# ── Stage 0a: Extract posed images ───────────────────────────────────────────

def stage_0a_extract_frames(scenes: list[str] | None = None) -> None:
    log_stage("Stage 0a - Extract posed images from .sens files")

    scans_root = SCANNET_DATA_ROOT / "scans"
    posed_root = SCANNET_DATA_ROOT / "posed_images"

    # Collect only scenes that actually have <scene>/<scene>.sens.
    available_sens_scenes: list[str] = []
    if scans_root.exists():
        for d in sorted(scans_root.iterdir()):
            if d.is_dir() and (d / f"{d.name}.sens").exists():
                available_sens_scenes.append(d.name)

    # Restrict extraction to requested scenes when provided.
    if scenes:
        missing_sens = [s for s in scenes if s not in available_sens_scenes]
        if missing_sens:
            log(f"  [WARN] Requested scenes missing .sens: {missing_sens}")
        target_scenes = [s for s in scenes if s in available_sens_scenes]
    else:
        target_scenes = available_sens_scenes

    if not target_scenes:
        log("  [WARN] No valid .sens scenes available for extraction.")
        log(f"         Expected files like: {scans_root}/<scene>/<scene>.sens")
        return

    missing_posed = [s for s in target_scenes if not (posed_root / s).exists()]
    if not missing_posed and posed_root.exists():
        log(f"  [SKIP] posed_images already exists for target scenes at {posed_root}")
        return

    # New upstream script scans every directory in ./scans. Build a filtered temporary
    # workspace so missing scene .sens files don't crash extraction.
    tmp_root = SCANNET_DATA_ROOT / ".stage0a_extract"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)

    tmp_scans = tmp_root / "scans"
    tmp_scans.mkdir(parents=True, exist_ok=True)
    for sid in target_scenes:
        _link_dir(tmp_scans / sid, scans_root / sid)

    posed_root.mkdir(parents=True, exist_ok=True)
    _link_dir(tmp_root / "posed_images", posed_root)

    nproc = 1 if os.name == "nt" else 4
    log(f"  Extracting posed images for {len(target_scenes)} scene(s) (frame_skip=20, nproc={nproc})...")
    rc = run_cmd(
        [sys.executable, str(VLM_GROUNDER_REPO / "data" / "scannet" / "tools" / "extract_posed_images.py"),
         "--frame_skip", "20", "--nproc", str(nproc)],
        cwd=tmp_root,
    )
    if rc != 0:
        raise RuntimeError(f"Stage 0a failed (exit {rc})")

    shutil.rmtree(tmp_root, ignore_errors=True)

    # Re-create symlink if posed_images was just created
    if not VG_POSED_DIR.exists():
        _link_dir(VG_POSED_DIR, posed_root)

    log("  [DONE] Posed images extracted.")

    # Sanity: verify each target scene has frames
    for sid in target_scenes:
        scene_dir = posed_root / sid
        if scene_dir.exists():
            n_jpg = len(list(scene_dir.glob("*.jpg")))
            n_pose = len(list(scene_dir.glob("*.txt"))) - 1  # subtract intrinsic.txt
            log(f"  [SANITY OK] {sid}: {n_jpg} frames, {max(n_pose,0)} poses")
        else:
            log(f"  [SANITY WARN] {sid}: posed_images directory not found at {scene_dir}")


# ── Stage 0b: Scene info pkl ──────────────────────────────────────────────────

def stage_0b_scene_info(scenes: list[str] | None = None) -> None:
    log_stage("Stage 0b - Generate scene info pkl")

    out_pkl = VG_INSTANCE_DATA / "scenes_train_val_info_w_images.pkl"
    if _pkl_covers_scenes(out_pkl, scenes, require_images_info=True):
        log(f"  [SKIP] scene info pkl already covers requested scenes: {out_pkl}")
        return

    all_scan_ids = sorted([d.name for d in VG_SCANS_DIR.iterdir() if d.is_dir()])
    if scenes:
        selected_ids = [sid for sid in scenes if sid in all_scan_ids]
    else:
        selected_ids = all_scan_ids

    def ensure_mesh_file(scene_id: str) -> bool:
        d = VG_SCANS_DIR / scene_id
        mesh_file = d / f"{scene_id}_vh_clean_2.ply"
        labels_mesh = d / f"{scene_id}_vh_clean_2.labels.ply"
        if mesh_file.exists():
            return True
        if labels_mesh.exists():
            try:
                try:
                    mesh_file.symlink_to(labels_mesh.name)
                except OSError:
                    shutil.copy2(labels_mesh, mesh_file)
                log(f"  [INFO] Created mesh symlink for {scene_id}: {mesh_file.name} -> {labels_mesh.name}")
            except FileExistsError:
                pass
            return mesh_file.exists()
        return False

    def has_scene_info_inputs(scene_id: str) -> bool:
        d = VG_SCANS_DIR / scene_id
        has_mesh = ensure_mesh_file(scene_id)
        required = [
            d / f"{scene_id}.aggregation.json",
            d / f"{scene_id}_vh_clean_2.0.010000.segs.json",
            d / f"{scene_id}.txt",
        ]
        return has_mesh and all(p.exists() for p in required)

    scan_ids = [sid for sid in selected_ids if has_scene_info_inputs(sid)]
    skipped = [sid for sid in selected_ids if sid not in scan_ids]
    if skipped:
        log(f"  [WARN] Skipping {len(skipped)} scene(s) missing Stage 0b inputs: {skipped}")

    if not scan_ids:
        raise RuntimeError(
            "Stage 0b failed: no valid scenes with required mesh/seg/agg/meta files "
            f"under {VG_SCANS_DIR}"
        )

    subset_file = VG_SCANNET_ROOT / "meta_data" / "scannet_subset.txt"
    subset_file.write_text("\n".join(scan_ids) + "\n", encoding="utf-8")

    scannet_cwd = VG_SCANNET_ROOT

    batch_loader = VG_SCANNET_ROOT / "tools" / "batch_load_scannet_data.py"
    batch_loader_text = batch_loader.read_text(encoding="utf-8", errors="ignore")
    batch_loader_cmd = [
        sys.executable,
        "tools/batch_load_scannet_data.py",
        "--output_folder",
        "scannet_instance_data",
        "--train_scannet_dir",
        "scans",
        "--train_scan_names_file",
        "meta_data/scannet_subset.txt",
    ]
    if "--num_workers" in batch_loader_text:
        num_workers = "1" if os.name == "nt" else "20"
        batch_loader_cmd.extend(["--num_workers", num_workers])
    else:
        log("  [INFO] batch_load_scannet_data.py does not support --num_workers; using upstream defaults.")

    rc = run_cmd(
        batch_loader_cmd,
        cwd=scannet_cwd,
    )
    if rc != 0:
        raise RuntimeError(f"Stage 0b failed (exit {rc})")

    # Add posed-image metadata expected by SceneInfoHandler.
    rc = run_cmd([sys.executable, "tools/update_info_file_with_images.py"], cwd=scannet_cwd)
    if rc != 0:
        raise RuntimeError(f"Stage 0b (update_info_file_with_images) failed (exit {rc})")

    _assert_file(out_pkl, "Stage 0b", min_bytes=1024)


# ── Stage 2: PATS exhaustive matching ────────────────────────────────────────

def stage_2_pats_matching(vg_csv_path: Path) -> None:
    log_stage("Stage 2 - PATS exhaustive matching (~20 min/scene, one-time)")

    out_pkl = VG_MATCH_DATA / "exhaustive_matching.pkl"
    with open(vg_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        scene_ids = sorted({row["scan_id"] for row in reader if row.get("scan_id")})

    if _pkl_covers_scenes(out_pkl, scene_ids, require_images_info=False):
        log(f"  [SKIP] PATS matching pkl already covers requested scenes: {out_pkl}")
        return

    pats_config = VLM_GROUNDER_REPO / "3rdparty" / "pats" / "configs" / "test_scannet.yaml"
    if not pats_config.exists():
        raise FileNotFoundError(f"PATS config not found: {pats_config}")

    # PATS uses a compiled C++ tensor_resize extension that requires libc10.so
    # from PyTorch's lib directory at runtime.
    torch_lib = Path(sys.executable).parent.parent / "lib" / "python3.10" / "site-packages" / "torch" / "lib"
    existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
    pats_env = {"LD_LIBRARY_PATH": f"{torch_lib}:{existing_ld}" if existing_ld else str(torch_lib)}
    # PATS can hit allocator fragmentation on shared GPUs; this lowers peak split size.
    pats_env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    rc = run_cmd(
        [sys.executable, "vlm_grounder/tools/exhaustive_matching.py",
         "--vg_file",      str(vg_csv_path.resolve()),
         "--config",       str(pats_config)],
        cwd=VLM_GROUNDER_REPO,
        env=pats_env,
    )
    if rc != 0:
        raise RuntimeError(f"Stage 2 failed (exit {rc})")

    produced = VG_MATCH_DATA / f"{vg_csv_path.stem}_exhaustive_matching.pkl"
    if produced.exists() and produced != out_pkl:
        try:
            if out_pkl.exists() or out_pkl.is_symlink():
                out_pkl.unlink()
            out_pkl.symlink_to(produced.name)
        except OSError:
            shutil.copy2(produced, out_pkl)

    _assert_file(out_pkl, "Stage 2", min_bytes=1024)


# ── Stage 3: Query analysis ───────────────────────────────────────────────────

def stage_3_query_analysis(
    vg_csv_path: Path, llm_backend: str, llm_model: str,
    vllm_port: int = VLLM_PORT,
) -> Path:
    log_stage("Stage 3 - Query analysis (parse utterances -> target class + attributes)")

    env = _llm_env(llm_backend, vllm_port, llm_model)
    stem = vg_csv_path.stem

    out_dir = VLM_GROUNDER_REPO / "outputs" / "query_analysis"
    before = {p.resolve() for p in out_dir.glob(f"{stem}*.csv")}

    rc = run_cmd(
        [sys.executable, "vlm_grounder/tools/query_analysis.py",
         "--vg_file",     str(vg_csv_path.resolve()),
         "--output_dir",  str(out_dir),
         "--llm_backend", llm_backend,
         "--llm_model",   llm_model],
        cwd=VLM_GROUNDER_REPO,
        env=env,
    )
    if rc != 0:
        raise RuntimeError(f"Stage 3 failed (exit {rc})")

    # VLM-Grounder appends a suffix to the CSV name.
    # Prefer files produced by this invocation to avoid stale outputs from older runs.
    candidates = sorted(out_dir.glob(f"{stem}*.csv"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"Stage 3 output CSV not found under {out_dir}")
    fresh = [p for p in candidates if p.resolve() not in before]
    result = fresh[-1] if fresh else candidates[-1]
    # Expected columns: pred_target_class, attributes, conditions (LLM parses the query)
    _assert_csv_rows(result, "Stage 3", min_rows=1, required_cols=["pred_target_class"])
    return result


# ── Stage 4: Instance detection ───────────────────────────────────────────────

def stage_4_detection(query_csv: Path) -> Path:
    log_stage("Stage 4 - Instance detection (YOLO-World, no API cost)")

    det_dir = VLM_GROUNDER_REPO / "outputs" / "image_instance_detector"
    scene_ids: list[str] = []
    with open(query_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid = (row.get("scan_id") or "").strip()
            if sid:
                scene_ids.append(sid)
    scene_ids = sorted(set(scene_ids))

    def _run_detector_once() -> Path:
        before = {p.resolve() for p in det_dir.rglob("detection.pkl")}
        rc = run_cmd(
            [sys.executable, "vlm_grounder/tools/image_instance_detector.py",
             "--vg_file",   str(query_csv),
             "--detector",  "yolo",
             "--chunk_size", "-1"],
            cwd=VLM_GROUNDER_REPO,
        )
        if rc != 0:
            raise RuntimeError(f"Stage 4 failed (exit {rc})")

        candidates = sorted(det_dir.rglob("detection.pkl"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError(f"Stage 4 detection.pkl not found under {det_dir}")
        fresh = [p for p in candidates if p.resolve() not in before]
        result = fresh[-1] if fresh else candidates[-1]
        _assert_file(result, "Stage 4", min_bytes=512)
        return result

    def _find_missing_image_keys(det_pkl: Path) -> dict[str, list[str]]:
        if not scene_ids:
            return {}
        try:
            with open(det_pkl, "rb") as f:
                det_infos = pickle.load(f)
        except Exception as exc:
            raise RuntimeError(f"Stage 4 detection.pkl unreadable: {det_pkl} ({type(exc).__name__}: {exc})")
        if not isinstance(det_infos, dict):
            raise RuntimeError(f"Stage 4 detection.pkl has unexpected format: {det_pkl}")

        missing: dict[str, list[str]] = {}
        for sid in scene_ids:
            scene_map = det_infos.get(sid, {})
            if not isinstance(scene_map, dict):
                scene_map = {}
            scene_keys = {str(k).replace("\\", "/") for k in scene_map.keys()}
            posed_dir = VG_POSED_DIR / sid
            if not posed_dir.exists():
                continue
            for img_path in sorted(posed_dir.glob("*.jpg")):
                rel = f"data/scannet/posed_images/{sid}/{img_path.name}"
                if rel not in scene_keys:
                    missing.setdefault(sid, []).append(rel)
        return missing

    result = _run_detector_once()
    missing = _find_missing_image_keys(result)
    if missing:
        total_missing = sum(len(v) for v in missing.values())
        log(
            f"  [WARN] Stage 4 detection cache is incomplete ({total_missing} missing image keys). "
            "Rebuilding detector cache once."
        )
        cache_root = det_dir / f"yolo_{query_csv.stem}"
        if cache_root.exists():
            shutil.rmtree(cache_root, ignore_errors=True)
        result = _run_detector_once()
        missing = _find_missing_image_keys(result)
        if missing:
            total_missing = sum(len(v) for v in missing.values())
            sample = []
            for sid, keys in missing.items():
                sample.extend(keys[:2])
                if len(sample) >= 8:
                    break
            raise RuntimeError(
                "Stage 4 coverage sanity failed after rebuild: detection.pkl is still missing "
                f"{total_missing} image keys. Example missing keys: {sample}"
            )
        log("  [OK] Stage 4 detection coverage sanity passed after cache rebuild.")
    return result


# ── Stage 5: View pre-selection ───────────────────────────────────────────────

def stage_5_view_selection(
    query_csv: Path,
    det_pkl: Path,
    llm_backend: str,
    llm_model: str,
    vllm_port: int = VLLM_PORT,
) -> Path:
    log_stage("Stage 5 - View pre-selection")

    env = _llm_env(llm_backend, vllm_port, llm_model)
    out_dir = VLM_GROUNDER_REPO / "outputs" / "query_analysis"
    before = {p.resolve() for p in out_dir.glob("*_with_images_selected*.csv")}
    # Upstream view_pre_selection defaults to sample_num=250; override so wrapper
    # processes all rows from the prepared CSV.
    with open(query_csv, newline="", encoding="utf-8") as f:
        total_queries = max(0, sum(1 for _ in csv.DictReader(f)))
    if total_queries == 0:
        raise RuntimeError(f"Stage 5 input CSV has no query rows: {query_csv}")

    rc = run_cmd(
        [sys.executable, "vlm_grounder/tools/view_pre_selection.py",
         "--vg_file",  str(query_csv),
         "--det_file", str(det_pkl),
         "--sample_num", str(total_queries),
         "--llm_backend", llm_backend,
         "--llm_model", llm_model],
        cwd=VLM_GROUNDER_REPO,
        env=env,
    )
    if rc != 0:
        raise RuntimeError(f"Stage 5 failed (exit {rc})")

    candidates = sorted(out_dir.glob("*_with_images_selected*.csv"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"Stage 5 output CSV not found under {out_dir}")
    fresh = [p for p in candidates if p.resolve() not in before]
    pool = fresh if fresh else candidates
    # Prefer the final full preselection CSV (non-topN), because *_topN.csv is an
    # intermediate artifact that can silently cap query coverage (e.g., top500).
    non_topn = [p for p in pool if not re.search(r"_top\d+$", p.stem)]
    result = non_topn[-1] if non_topn else pool[-1]
    if not non_topn:
        log(f"  [WARN] Stage 5 selected topN CSV (no non-topN file found): {result.name}")
    # Expected: matched_image_ids_confidence* columns + instance_pkl_path
    _assert_csv_rows(result, "Stage 5", min_rows=1,
                     required_cols=["matched_image_ids_confidence0.1", "instance_pkl_path"])
    return result


# ── Stage 6: Visual grounding ─────────────────────────────────────────────────

def stage_6_visual_grounding(
    view_csv: Path, det_pkl: Path, run_id: str,
    llm_backend: str, llm_model: str,
    vllm_port: int = VLLM_PORT,
) -> Path:
    log_stage("Stage 6 - Visual grounding (main inference, LLM calls)")

    scene_info_pkl = VG_INSTANCE_DATA / "scenes_train_val_info_w_images.pkl"
    match_pkl      = VG_MATCH_DATA    / "exhaustive_matching.pkl"
    out_dir        = VLM_GROUNDER_REPO / "outputs" / "visual_grounding"

    env = _llm_env(llm_backend, vllm_port, llm_model)
    # Keep Stage 6 close to upstream VLM-Grounder defaults. In this codebase,
    # gpt_max_input_images controls how many selected views are stitched into
    # each grid image; lowering it increases the number of image messages and
    # can worsen vLLM context pressure.
    gpt_max_input_images = "6"
    max_fallback_images = "12"
    image_det_confidence = "0.2"
    log(
        "  [INFO] Using upstream-like Stage 6 settings "
        f"(gpt_max_input_images={gpt_max_input_images}, image_det_confidence={image_det_confidence})."
    )

    visual_grounder_py = VLM_GROUNDER_REPO / "vlm_grounder" / "grounder" / "visual_grouder.py"
    supports_max_fallback = (
        visual_grounder_py.exists()
        and "--max_fallback_images" in visual_grounder_py.read_text(encoding="utf-8", errors="ignore")
    )

    cmd = [
        sys.executable, "vlm_grounder/grounder/visual_grouder.py",
        "--from_scratch",
        "--vg_file_path",       str(view_csv),
        "--scene_info_path",    str(scene_info_pkl),
        "--det_info_path",      str(det_pkl),
        "--matching_info_path", str(match_pkl),
        "--output_dir",         str(out_dir),
        "--exp_name",           run_id,
        "--prompt_version",     "3",
        "--openaigpt_type",     llm_model,
        "--gpt_max_input_images", gpt_max_input_images,
        "--image_det_confidence", image_det_confidence,
        "--use_point_prompt",
        "--dynamic_stitching",
        "--online_detector",    "yolo",
    ]
    if supports_max_fallback:
        cmd.extend(["--max_fallback_images", max_fallback_images])
    else:
        log("  [INFO] visual_grouder.py does not support --max_fallback_images; skipping that flag.")

    rc = run_cmd(
        cmd,
        cwd=VLM_GROUNDER_REPO,
        env=env,
    )
    if rc != 0:
        raise RuntimeError(f"Stage 6 failed (exit {rc})")

    run_dir = out_dir / run_id
    candidates = sorted(run_dir.rglob("*_results.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"Stage 6 results JSON not found under: {run_dir}")
    results_json = candidates[-1]
    # gpt_pred_bbox is the predicted 3D bbox — may be at top level or inside eval_result
    _assert_json_list(results_json, "Stage 6", min_items=1, required_keys=["gpt_pred_bbox"])
    return results_json


# ── LLM env helper ────────────────────────────────────────────────────────────

def _llm_env(backend: str, vllm_port: int = VLLM_PORT, llm_model: str | None = None) -> dict:
    """
    VLM-Grounder only speaks the OpenAI API protocol.
    Gemini, Ollama, and vLLM all offer OpenAI-compatible endpoints, so we
    redirect via OPENAI_API_KEY + OPENAI_BASE_URL — no changes to VLM-Grounder needed.
    """
    if backend == "openai":
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise EnvironmentError("OPENAI_API_KEY not set. Export it or use --llm-backend gemini/ollama/vllm")
        env = {"OPENAI_API_KEY": key}
        if llm_model:
            env["OPENAI_MODEL"] = llm_model
        return env
    elif backend == "gemini":
        key = os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise EnvironmentError(
                "GEMINI_API_KEY not set. "
                "Get a free key at aistudio.google.com and add to ~/.bashrc"
            )
        env = {
            "OPENAI_API_KEY":  key,
            "OPENAI_BASE_URL": "https://generativelanguage.googleapis.com/v1beta/openai/",
        }
        if llm_model:
            env["OPENAI_MODEL"] = llm_model
        return env
    elif backend == "ollama":
        env = {
            "OPENAI_API_KEY":  "ollama",
            "OPENAI_BASE_URL": "http://localhost:11434/v1",
        }
        if llm_model:
            env["OPENAI_MODEL"] = llm_model
        return env
    elif backend == "vllm":
        env = {
            "OPENAI_API_KEY":  "vllm",
            "OPENAI_BASE_URL": f"http://localhost:{vllm_port}/v1",
        }
        if llm_model:
            env["OPENAI_MODEL"] = llm_model
        return env
    else:
        raise ValueError(f"Unknown llm_backend: {backend}")


# ── vLLM server lifecycle ─────────────────────────────────────────────────────

def _tail_lines(path: Path, num_lines: int = 80) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(errors="ignore").splitlines()[-num_lines:]


def _extract_vllm_root_cause(log_path: Path) -> str | None:
    """Return the most useful failure line from vLLM startup logs."""
    lines = _tail_lines(log_path, num_lines=200)
    if not lines:
        return None
    priority_markers = [
        "ValueError:",
        "RuntimeError:",
        "Engine core initialization failed",
        "ERROR",
    ]
    for marker in priority_markers:
        for ln in reversed(lines):
            if marker in ln:
                return ln.strip()
    return lines[-1].strip() if lines else None


def _contains_any(text: str | None, patterns: list[str]) -> bool:
    if not text:
        return False
    t = text.lower()
    return any(p.lower() in t for p in patterns)


def _suggest_next_vllm_settings(
    root_cause: str | None,
    current_gpu_mem_util: float,
    current_max_model_len: int | None,
) -> tuple[float, int | None, str] | None:
    """Suggest one adaptive retry setting for known startup failures."""
    if _contains_any(root_cause, ["max seq len", "kv cache", "estimated maximum model length"]):
        if current_max_model_len is None:
            return current_gpu_mem_util, 4096, "limit max model len to 4096"
        if current_max_model_len > 4096:
            return current_gpu_mem_util, 4096, "reduce max model len to 4096"
    if _contains_any(root_cause, ["free memory on device", "desired gpu memory utilization"]):
        lowered = max(0.35, round(current_gpu_mem_util - 0.10, 2))
        if lowered < current_gpu_mem_util:
            return lowered, current_max_model_len, f"lower gpu-memory-utilization to {lowered:.2f}"
    return None


def start_vllm_server(
    model: str = VLLM_MODEL,
    port: int = VLLM_PORT,
    quantization: "str | None" = VLLM_QUANTIZATION,
    conda_env: str = VLLM_CONDA_ENV,
    gpu_memory_utilization: float = VLLM_GPU_MEM_UTIL,
    max_model_len: int | None = VLLM_MAX_MODEL_LEN,
    max_num_seqs: int = VLLM_MAX_NUM_SEQS,
    max_num_batched_tokens: int = VLLM_MAX_BATCHED_TOKENS,
    startup_retries: int = VLLM_STARTUP_RETRIES,
) -> "subprocess.Popen | None":
    """
    Launch a vLLM OpenAI-compatible API server as a background subprocess.

    vLLM runs in a separate conda env (vllm_server) because it requires
    CUDA 11.8+, incompatible with the vlm_grounder env's PyTorch+cu117 build
    which is pinned for PATS/PyTorch3D ABI compatibility.

    Args:
        model:                  HuggingFace model ID (e.g. "google/gemma-4-E4B-it")
        port:                   Port to listen on
        quantization:           None = BF16 full precision (default)
                                "bitsandbytes" = INT4 via bitsandbytes
                                "awq" = AWQ pre-quantized weights
        conda_env:              Conda environment name with vLLM installed
        gpu_memory_utilization: Fraction of GPU VRAM for vLLM (0.0–1.0).
                                Lower this when sharing the GPU (e.g. 0.40 ≈ 9.8 GB
                                on a 24 GB card, leaving room for VLM-Grounder + TOD).

    Returns:
        Popen handle to the server process — caller must call terminate() when done.
        Returns None if a server was already responding on the given port.
    """
    import time
    import urllib.request

    health_url = f"http://localhost:{port}/v1/models"

    # Return early if server already responding
    try:
        urllib.request.urlopen(health_url, timeout=2)
        log(f"  [OK] vLLM server already running on port {port} - reusing")
        return None
    except Exception:
        pass

    log_path = PROJECT / "outputs" / "vllm_server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    total_attempts = startup_retries + 1
    cur_gpu_mem_util = gpu_memory_utilization
    cur_max_model_len = max_model_len

    for attempt_idx in range(total_attempts):
        attempt_no = attempt_idx + 1
        quant_label = quantization or "BF16 (full precision)"
        log(f"  startup attempt {attempt_no}/{total_attempts}")
        log(f"  model        : {model}")
        log(f"  port         : {port}")
        log(f"  quantization : {quant_label}")
        log(f"  gpu-mem-util : {cur_gpu_mem_util:.2f}  ({cur_gpu_mem_util*100:.0f}% of VRAM)")
        log(f"  max-model-len: {cur_max_model_len if cur_max_model_len is not None else 'auto'}")
        log(f"  conda env    : {conda_env}")
        log()

        cmd = [
            "conda", "run", "-n", conda_env, "--no-capture-output",
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model,
            "--port", str(port),
            "--trust-remote-code",
            "--gpu-memory-utilization", str(cur_gpu_mem_util),
            "--max-num-seqs", str(max_num_seqs),
            "--max-num-batched-tokens", str(max_num_batched_tokens),
        ]
        if quantization:
            cmd += ["--quantization", quantization]
        if cur_max_model_len is not None:
            cmd += ["--max-model-len", str(cur_max_model_len)]

        log(f"  Server log   : {log_path}")
        log(f"  cmd: {' '.join(cmd)}")
        log()

        log_fh = open(log_path, "w")
        proc = subprocess.Popen(cmd, stdout=log_fh, stderr=log_fh)

        log("  Polling for server readiness (up to 3 min) ...")
        deadline = time.time() + 180
        while time.time() < deadline:
            try:
                urllib.request.urlopen(health_url, timeout=2)
                log("  [OK] vLLM server is ready")
                return proc
            except Exception:
                if proc.poll() is not None:
                    break
                time.sleep(5)

        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=30)
        log_fh.close()

        root_cause = _extract_vllm_root_cause(log_path)
        if root_cause:
            log(f"  [vLLM root cause] {root_cause}")

        if attempt_no >= total_attempts:
            break

        suggestion = _suggest_next_vllm_settings(root_cause, cur_gpu_mem_util, cur_max_model_len)
        if suggestion is None:
            break
        cur_gpu_mem_util, cur_max_model_len, reason = suggestion
        log(f"  [RETRY] Applying adaptive fallback: {reason}")
        log()

    raise RuntimeError(
        f"vLLM server did not become ready after {total_attempts} attempt(s).\n"
        f"  Check log: {log_path}\n"
        f"  Verify conda env '{conda_env}' has vLLM installed:\n"
        f"    conda activate {conda_env} && pip install vllm\n"
        f"  For INT4 (bitsandbytes): pip install bitsandbytes\n"
        f"  Tip: override with --vllm-max-model-len and/or --vllm-gpu-memory-utilization"
    )


# ── Output conversion: VLM-Grounder → schema v1 ───────────────────────────────

def convert_results_to_schema_v1(
    results_json_path: Path,
    input_csv_path: Path,
    out_dir: Path,
    run_id: str,
) -> int:
    """
    Parse VLM-Grounder results.json and write one schema v1 reconstruction
    JSON per query under out_dir/reconstruction/<scene_id>/<sample_id>.json.

    Returns number of files written.
    """
    with open(results_json_path) as f:
        results = json.load(f)

    # Build query_id lookup maps from the original input CSV.
    # VLM-Grounder outputs may omit query_id and may use `query` instead of `utterance`.
    utterance_to_qid: dict[tuple[str, str], str] = {}
    target_to_qid: dict[tuple[str, str], str] = {}
    with open(input_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scene = row["scan_id"]
            key = (scene, row["utterance"].strip())
            utterance_to_qid[key] = row["query_id"]
            target_id = str(row.get("target_id", "")).strip()
            if target_id:
                target_to_qid[(scene, target_id)] = row["query_id"]

    # ScanNet camera intrinsics — confirmed identical across all dev-mini scenes
    INTRINSICS = {
        "fx": 1170.187988, "fy": 1170.187988,
        "cx": 647.75,      "cy": 483.75,
        "color_width": 1296, "color_height": 968,
        "depth_width": 640,  "depth_height": 480,
    }
    DEPTH_META = {"depth_scale": 1000.0, "depth_trunc": 3.0}

    written = 0
    errors  = 0
    skipped_rows: list[dict[str, object]] = []

    # VLM-Grounder can save as list OR wrapped dict: {accuracy_*, results:[...]}.
    if isinstance(results, list):
        items = results
    elif isinstance(results, dict) and isinstance(results.get("results"), list):
        items = results["results"]
    else:
        items = [v for v in results.values() if isinstance(v, dict)] if isinstance(results, dict) else []

    for item in items:
        scene_id = item.get("scene_id", "")
        utterance = (item.get("utterance") or item.get("query") or "").strip()
        # gpt_pred_bbox may be at top level (some VLM-Grounder versions)
        # or nested inside eval_result (confirmed from repo source)
        eval_result = item.get("eval_result") or {}
        bbox_raw = item.get("gpt_pred_bbox") or eval_result.get("gpt_pred_bbox")

        if not scene_id or bbox_raw is None:
            log(f"  [SKIP] missing scene_id or gpt_pred_bbox: {item}")
            errors += 1
            skipped_rows.append(
                {
                    "reason": "missing_scene_id_or_gpt_pred_bbox",
                    "scene_id": scene_id or None,
                    "target_id": item.get("target_id"),
                    "query_id": item.get("query_id"),
                    "utterance": utterance,
                    "pred_target_class": item.get("pred_target_class"),
                    "gpt_pred_image_id": item.get("gpt_pred_image_id"),
                    "bbox_index": item.get("bbox_index"),
                }
            )
            continue

        target_id = str(item.get("target_id", "")).strip()
        query_id = (
            item.get("query_id")
            or utterance_to_qid.get((scene_id, utterance))
            or target_to_qid.get((scene_id, target_id))
        )
        if not query_id:
            log(f"  [WARN] Could not resolve query_id for {scene_id!r} / {utterance!r}")
            errors += 1
            skipped_rows.append(
                {
                    "reason": "unresolved_query_id",
                    "scene_id": scene_id,
                    "target_id": item.get("target_id"),
                    "query_id": item.get("query_id"),
                    "utterance": utterance,
                    "pred_target_class": item.get("pred_target_class"),
                    "gpt_pred_image_id": item.get("gpt_pred_image_id"),
                    "bbox_index": item.get("bbox_index"),
                }
            )
            continue

        # gpt_pred_bbox = [cx, cy, cz, dx, dy, dz]
        cx, cy, cz, dx, dy, dz = [float(v) for v in bbox_raw]
        sample_id = f"{scene_id}__{query_id}"

        record = {
            "schema_version":     "v1",
            "run_id":             run_id,
            "sample_id":          sample_id,
            "scene_id":           scene_id,
            "query_id":           str(query_id),
            "predicted_bbox_3d":  {
                "center": [cx, cy, cz],
                "size":   [dx, dy, dz],
            },
            "predicted_point_cloud": None,
            "camera_intrinsics":  INTRINSICS,
            "depth_metadata":     DEPTH_META,
            "coordinate_frame":   "scannet_world",
        }

        dest = out_dir / "reconstruction" / scene_id / f"{sample_id}.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(record, indent=2))
        written += 1

    skipped_dir = out_dir / "reconstruction"
    skipped_dir.mkdir(parents=True, exist_ok=True)
    skipped_json_path = skipped_dir / "_skipped_queries.json"
    skipped_txt_path = skipped_dir / "_skipped_queries.txt"
    skipped_json_path.write_text(json.dumps(skipped_rows, indent=2), encoding="utf-8")
    skipped_lines = [
        f"written={written}",
        f"skipped={errors}",
    ]
    for row in skipped_rows:
        skipped_lines.append(
            " | ".join(
                [
                    f"reason={row.get('reason')}",
                    f"scene_id={row.get('scene_id')}",
                    f"target_id={row.get('target_id')}",
                    f"query_id={row.get('query_id')}",
                    f"pred_target_class={row.get('pred_target_class')}",
                    f"utterance={row.get('utterance')}",
                ]
            )
        )
    skipped_txt_path.write_text("\n".join(skipped_lines) + "\n", encoding="utf-8")

    log(f"  [DONE] {written} schema v1 files written, {errors} skipped")
    log(f"  [INFO] Skipped-query summary: {skipped_json_path}")
    return written


def save_run_artifacts(
    out_dir: Path,
    run_id: str,
    manifest_path: Path,
    scenes: list[str] | None,
    llm_backend: str,
    llm_model: str,
    skip_one_time: bool,
    results_json_path: Path,
    n_queries: int,
    n_written: int,
) -> None:
    """Persist a concise, self-contained run record inside outputs/run_<id>/."""
    raw_dir = out_dir / "raw_vlm_grounder"
    raw_dir.mkdir(parents=True, exist_ok=True)

    raw_results_copy = raw_dir / results_json_path.name
    shutil.copy2(results_json_path, raw_results_copy)
    source_vg_run_dir = results_json_path.parent.parent if results_json_path.parent != raw_dir else results_json_path.parent

    vllm_log = PROJECT / "outputs" / "vllm_server.log"
    vllm_log_copy = None
    if vllm_log.exists():
        vllm_log_copy = raw_dir / vllm_log.name
        shutil.copy2(vllm_log, vllm_log_copy)

    session_lines = [
        f"run_id: {run_id}",
        f"out_dir: {out_dir}",
        f"manifest: {manifest_path}",
        f"scenes: {scenes or 'all dev-mini'}",
        f"llm_backend: {llm_backend}",
        f"llm_model: {llm_model}",
        f"skip_one_time: {skip_one_time}",
        f"vg_input_csv: {out_dir / 'vg_input.csv'}",
        f"source_vg_run_dir: {source_vg_run_dir}",
        f"source_results_json: {results_json_path}",
        f"raw_results_json: {raw_results_copy}",
        f"reconstruction_dir: {out_dir / 'reconstruction'}",
        f"skipped_queries_json: {out_dir / 'reconstruction' / '_skipped_queries.json'}",
        f"n_queries_in_manifest_subset: {n_queries}",
        f"n_reconstruction_json_written: {n_written}",
        f"vllm_server_log: {vllm_log_copy or 'not_copied'}",
    ]
    (out_dir / "session_info.txt").write_text("\n".join(session_lines) + "\n", encoding="utf-8")

    summary = {
        "run_id": run_id,
        "out_dir": str(out_dir),
        "manifest": str(manifest_path),
        "scenes": scenes or "all dev-mini",
        "llm_backend": llm_backend,
        "llm_model": llm_model,
        "skip_one_time": skip_one_time,
        "vg_input_csv": str(out_dir / "vg_input.csv"),
        "source_vg_run_dir": str(source_vg_run_dir),
        "source_results_json": str(results_json_path),
        "raw_results_json": str(raw_results_copy),
        "reconstruction_dir": str(out_dir / "reconstruction"),
        "skipped_queries_json": str(out_dir / "reconstruction" / "_skipped_queries.json"),
        "n_queries_in_manifest_subset": n_queries,
        "n_reconstruction_json_written": n_written,
        "vllm_server_log": str(vllm_log_copy) if vllm_log_copy else None,
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    summary_lines = [
        f"VLM-Grounder wrapper run summary for {run_id}",
        f"Queries converted: {n_queries}",
        f"Reconstruction JSONs written: {n_written}",
        f"Run directory: {out_dir}",
        f"Reconstruction output: {out_dir / 'reconstruction'}",
        f"Skipped-query summary: {out_dir / 'reconstruction' / '_skipped_queries.json'}",
        f"Raw Step 6 result copy: {raw_results_copy}",
    ]
    (out_dir / "run_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run VLM-Grounder baseline on dev-mini")
    parser.add_argument("--manifest",    default=str(PROJECT / "data" / "dev_mini_manifest.json"),
                        help="Path to dev_mini_manifest.json")
    parser.add_argument("--out",         required=True,
                        help="Output directory (e.g. outputs/run_<id>)")
    parser.add_argument("--run_id",      required=True,
                        help="Run identifier string")
    parser.add_argument("--scenes",      nargs="+", default=None,
                        help="Subset of scene IDs to run (default: all dev-mini)")
    parser.add_argument("--llm-backend", default=DEFAULT_LLM_BACKEND,
                        choices=["openai", "ollama", "gemini", "vllm"],
                        help="LLM backend: openai | ollama | gemini | vllm")
    parser.add_argument("--llm-model",   default=None,
                        help="Override LLM model name")
    parser.add_argument("--skip-one-time", action="store_true",
                        help="Skip Stages 0a/0b/2 (already done on this server)")
    # vLLM options (only used when --llm-backend vllm)
    parser.add_argument("--vllm-model",        default=None,
                        help=f"HuggingFace model ID for vLLM (default: {VLLM_MODEL})")
    parser.add_argument("--vllm-port",         type=int, default=VLLM_PORT,
                        help=f"vLLM server port (default: {VLLM_PORT})")
    parser.add_argument("--vllm-quantization", default=None,
                        choices=["bitsandbytes", "awq"],
                        help="vLLM quantization: bitsandbytes=INT4, awq=AWQ (default: None=BF16)")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=VLLM_GPU_MEM_UTIL,
                        metavar="FRAC",
                        help=(
                            f"Fraction of GPU VRAM reserved for vLLM (default: {VLLM_GPU_MEM_UTIL}). "
                            "Lower when sharing GPU - 0.40 ~= 9.8 GB on 24 GB, "
                            "leaving room for VLM-Grounder + other processes."
                        ))
    parser.add_argument("--vllm-max-model-len", type=int, default=VLLM_MAX_MODEL_LEN,
                        help=(
                            "Optional cap for vLLM context length. "
                            "Use lower values (e.g. 4096) to reduce KV-cache pressure on shared GPUs."
                        ))
    parser.add_argument("--vllm-max-num-seqs", type=int, default=VLLM_MAX_NUM_SEQS,
                        help="Maximum concurrent sequences for vLLM (default: conservative shared-GPU value).")
    parser.add_argument("--vllm-max-num-batched-tokens", type=int, default=VLLM_MAX_BATCHED_TOKENS,
                        help="Maximum batched prefill tokens for vLLM (default: conservative shared-GPU value).")
    parser.add_argument("--vllm-startup-retries", type=int, default=VLLM_STARTUP_RETRIES,
                        help="Number of adaptive retries if vLLM startup fails with known memory constraints.")
    args = parser.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve LLM model
    if args.llm_model:
        llm_model = args.llm_model
    elif args.llm_backend == "openai":
        llm_model = DEFAULT_LLM_MODEL
    elif args.llm_backend == "ollama":
        llm_model = OLLAMA_MODEL
    elif args.llm_backend == "vllm":
        llm_model = args.vllm_model or VLLM_MODEL
    else:
        llm_model = GEMINI_MODEL

    log(SEP)
    log("  VLM-Grounder Wrapper")
    log(SEP)
    log(f"  run_id      : {args.run_id}")
    log(f"  out_dir     : {out_dir}")
    log(f"  manifest    : {args.manifest}")
    log(f"  scenes      : {args.scenes or 'all dev-mini'}")
    log(f"  llm_backend : {args.llm_backend}")
    log(f"  llm_model   : {llm_model}")
    if args.llm_backend == "vllm":
        log(f"  vllm_port   : {args.vllm_port}")
        log(f"  vllm_quant  : {args.vllm_quantization or 'BF16 (full precision)'}")
        log(f"  gpu_mem_util: {args.vllm_gpu_memory_utilization:.2f}")
        log(f"  max_model_len: {args.vllm_max_model_len if args.vllm_max_model_len is not None else 'auto'}")
        log(f"  max_num_seqs : {args.vllm_max_num_seqs}")
        log(f"  max_batched_toks: {args.vllm_max_num_batched_tokens}")
        log(f"  startup_retries : {args.vllm_startup_retries}")
    log()

    # Pre-flight checks
    if not VLM_GROUNDER_REPO.exists():
        log(f"  [ERROR] VLM-Grounder repo not found: {VLM_GROUNDER_REPO}")
        log("          Clone it: cd scripts && git clone https://github.com/InternRobotics/VLM-Grounder.git vlm-grounder-repo")
        sys.exit(1)

    manifest_path = ensure_manifest_for_run(Path(args.manifest), args.scenes)
    if not manifest_path.exists():
        log(f"  [ERROR] Manifest not found after preparation: {manifest_path}")
        sys.exit(1)
    if str(manifest_path) != str(args.manifest):
        log(f"  [OK] Using autogenerated manifest: {manifest_path}")
        args.manifest = str(manifest_path)
    write_runtime_snapshot(out_dir, args.scenes, manifest_path)

    # ── Step 1: Convert manifest → VLM-Grounder CSV
    log_stage("Step 1 - Convert manifest to VLM-Grounder CSV")
    vg_csv = out_dir / "vg_input.csv"
    from convert_manifest_to_vg_csv import convert as convert_csv  # noqa: E402
    n_queries = convert_csv(args.manifest, str(vg_csv), args.scenes)
    log(f"  {n_queries} queries -> {vg_csv}")

    # ── Patch VLM-Grounder source files before any stage runs
    patch_gdino_api_key()
    patch_openai_setup()
    patch_visual_grounder_json_retry()

    # ── Data layout setup
    ensure_vg_data_layout()

    # ── One-time stages (skip with --skip-one-time if already done on this server)
    if not args.skip_one_time:
        ensure_scannet_scene_assets(args.scenes)
        stage_0a_extract_frames(args.scenes)
        stage_0b_scene_info(args.scenes)
        stage_2_pats_matching(vg_csv)
    else:
        log(); log("  [SKIP] One-time stages (--skip-one-time set)")

    # ── Start vLLM server if needed (runs in background, terminated on exit)
    vllm_proc = None
    if args.llm_backend == "vllm":
        log_stage("Starting vLLM server")
        vllm_proc = start_vllm_server(
            model=llm_model,
            port=args.vllm_port,
            quantization=args.vllm_quantization,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            max_model_len=args.vllm_max_model_len,
            max_num_seqs=args.vllm_max_num_seqs,
            max_num_batched_tokens=args.vllm_max_num_batched_tokens,
            startup_retries=args.vllm_startup_retries,
        )

    # ── Per-run stages (wrapped so vLLM server is always stopped on error/exit)
    # Between stages the sentinel file is checked: `touch outputs/STOP` stops cleanly.
    try:
        check_stop_sentinel()
        query_csv  = stage_3_query_analysis(vg_csv, args.llm_backend, llm_model, args.vllm_port)
        check_stop_sentinel()
        det_pkl    = stage_4_detection(query_csv)
        check_stop_sentinel()
        view_csv   = stage_5_view_selection(
            query_csv,
            det_pkl,
            args.llm_backend,
            llm_model,
            args.vllm_port,
        )
        check_stop_sentinel()
        results_json = stage_6_visual_grounding(
            view_csv, det_pkl, args.run_id, args.llm_backend, llm_model, args.vllm_port
        )
    finally:
        if vllm_proc is not None:
            log()
            log("  Stopping vLLM server ...")
            vllm_proc.terminate()
            vllm_proc.wait()
            log("  [OK] vLLM server stopped")

    # ── Convert to schema v1
    log_stage("Step 7 - Convert to schema v1 reconstruction format")
    n_written = convert_results_to_schema_v1(results_json, vg_csv, out_dir, args.run_id)
    save_run_artifacts(
        out_dir=out_dir,
        run_id=args.run_id,
        manifest_path=Path(args.manifest),
        scenes=args.scenes,
        llm_backend=args.llm_backend,
        llm_model=llm_model,
        skip_one_time=args.skip_one_time,
        results_json_path=results_json,
        n_queries=n_queries,
        n_written=n_written,
    )

    log()
    log(SEP)
    log("  DONE")
    log(SEP)
    log(f"  {n_written}/{n_queries} reconstruction JSONs written to {out_dir}/reconstruction/")
    log(f"  Saved run summary: {out_dir / 'run_summary.json'}")
    log(f"  Run evaluator:  python scripts/evaluate.py --run {out_dir} \\")
    log(f"                      --nr3d data/ReferIt3D/nr3d.csv \\")
    log(f"                      --scannet data/ScanNet/scans")


if __name__ == "__main__":
    # Make sure scripts/ is on the path so convert_manifest_to_vg_csv is importable
    sys.path.insert(0, str(Path(__file__).parent))
    main()
