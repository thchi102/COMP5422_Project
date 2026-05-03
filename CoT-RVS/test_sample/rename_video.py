import os
import re
from pathlib import Path
frame_dir = Path("26d141ec-f952-3908-b4cc-ae359377424e")
# Match names like: ring_front_center_315970956442504176.jpg
pattern = re.compile(r".*_(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)
files = []
for p in frame_dir.iterdir():
    if p.is_file():
        m = pattern.match(p.name)
        if m:
            ts = int(m.group(1))  # timestamp suffix
            ext = m.group(2).lower()
            files.append((ts, p, ext))
if not files:
    raise RuntimeError(f"No matching frame files found in {frame_dir}")
# Sort by timestamp extracted from filename
files.sort(key=lambda x: x[0])
# Keep original extension family if you want; SAM2 usually uses jpg frames
target_ext = "jpg"
# Pass 1: rename to temporary names
temp_paths = []
for idx, (_, old_path, _) in enumerate(files):
    tmp = frame_dir / f"__tmp__{idx:06d}.{target_ext}"
    old_path.rename(tmp)
    temp_paths.append(tmp)
# Pass 2: rename to final numeric names: 000000.jpg, 000001.jpg, ...
for idx, tmp in enumerate(temp_paths):
    final = frame_dir / f"{idx:06d}.{target_ext}"
    tmp.rename(final)
print(f"Renamed {len(temp_paths)} frames in {frame_dir}")