import os
from PIL import Image, ImageFont, ImageDraw
from pathlib import Path
from skvideo import io
import ast
import cv2
import numpy as np

def _load_number_font(size):
    # Try common scalable fonts first. PIL default font is bitmap-based and tiny.
    for font_name in [
        "DejaVuSans-Bold.ttf",
        "DejaVuSans.ttf",
        "NotoSans-Regular.ttf",
        "NotoSansMath-Regular.ttf",
    ]:
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            continue
    return ImageFont.load_default()

def save_video(
    images,
    path,
):

    # Create the parent directory if it doesn't already exist.
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)

    if len(images) == 0:
        raise ValueError("save_video received an empty image list.")

    first = np.asarray(images[0])
    if first.ndim != 3 or first.shape[2] != 3:
        raise ValueError(
            f"Expected RGB frames with shape (H, W, 3), got {first.shape}."
        )
    height, width = first.shape[:2]

    normalized = []
    for idx, frame in enumerate(images):
        arr = np.asarray(frame)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Frame {idx} has invalid shape {arr.shape}.")
        if arr.shape[:2] != (height, width):
            raise ValueError(
                f"Frame {idx} shape {arr.shape[:2]} does not match first frame {(height, width)}."
            )
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        normalized.append(arr)

    # Try ffmpeg first; if the pipe fails, fallback to OpenCV writer.
    try:
        writer = io.FFmpegWriter(
            str(path),
            outputdict={
                "-pix_fmt": "yuv420p",
                "-crf": "21",
                "-vf": "setpts=PTS",
            },
        )
        for frame in normalized:
            writer.writeFrame(frame)
        writer.close()
        return
    except Exception as ffmpeg_err:
        try:
            writer.close()
        except Exception:
            pass

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        cv_writer = cv2.VideoWriter(str(path), fourcc, 10.0, (width, height))
        if not cv_writer.isOpened():
            raise RuntimeError(
                f"Failed to initialize both FFmpegWriter and OpenCV VideoWriter. "
                f"FFmpeg error: {ffmpeg_err}"
            )
        for frame in normalized:
            # OpenCV expects BGR.
            cv_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv_writer.release()
    
def merge_keyframe(args,video_name,images,max_size=3200):
    if len(images) == 0:
        raise ValueError("merge_keyframe received an empty image list.")

    width, height = images[0].size
    max_images_per_column = 8
    num_images = len(images)
    num_rows = min(max_images_per_column, num_images)
    num_cols = (num_images + max_images_per_column - 1) // max_images_per_column

    index_spacing = max(64, min(width, height) // 3)
    spacing = max(8, min(width, height) // 30)
    boundary = 10

    cell_width = index_spacing + width
    cell_height = height
    total_width = num_cols * cell_width + (num_cols - 1) * spacing + 2 * boundary
    total_height = num_rows * cell_height + (num_rows - 1) * spacing + 2 * boundary

    final_image = Image.new("RGB", (total_width, total_height), (255, 255, 255))
    draw = ImageDraw.Draw(final_image)
    font = _load_number_font(max(40, min(width, height) // 4))

    for idx, img in enumerate(images):
        row = idx % max_images_per_column
        col = idx // max_images_per_column

        x_offset = boundary + col * (cell_width + spacing)
        y_offset = boundary + row * (cell_height + spacing)

        # Number strip on the left side of each frame.
        draw.rectangle(
            [(x_offset, y_offset), (x_offset + index_spacing - 1, y_offset + height - 1)],
            fill=(255, 255, 255),
            outline=(0, 0, 0),
            width=2,
        )
        label = str(idx + 1)
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        text_x = x_offset + (index_spacing - text_w) // 2
        text_y = y_offset + (height - text_h) // 2
        draw.text(
            (text_x, text_y),
            label,
            fill=(0, 0, 0),
            font=font,
            stroke_width=3,
            stroke_fill=(255, 255, 255),
        )

        final_image.paste(img, (x_offset + index_spacing, y_offset))

    result_image = final_image
    if result_image.width > result_image.height:
        if result_image.width > max_size:
            scale_factor = max_size / result_image.width
            new_size = (max_size, int(result_image.height * scale_factor))
            result_image = result_image.resize(new_size, Image.Resampling.LANCZOS)
    else:
        if result_image.height > max_size:
            scale_factor = max_size / result_image.height
            new_size = (int(result_image.width * scale_factor), max_size)
            result_image = result_image.resize(new_size, Image.Resampling.LANCZOS)
    save_path = os.path.join(args.output_dir,"keyframes",f"{video_name}.jpg")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    result_image.save(save_path)
    return save_path

def preprocess_prompt(query,num_keyframes):
    
    return f"""You will act as a keyframe selection agent for a video reasoning task. During each inference, you will be given a grid image that contains multiple keyframes sampled from a long video. The keyframes are aligned from top to down or from left to right, following their temporal order. You will also be given a user query. You must identify exactly ONE target: the single object or instance in the scene that best matches the user query. Do not return multiple different targets. If several candidates could match, compare them in your reasoning and select the one that best fits the query (specificity, attributes, and context). You need to think in chain of thoughts to analyze the keyframes and choose the best keyframe for that single chosen object—one where a segmentation model can find the object with minimal ambiguity. Your chain of thoughts should begin with what can be seen in each keyframe, which candidates could match the query, and why one choice is the best match. Some objects may be seriously obscured or blocked; some may be camouflaged. Analyze each frame separately, then compare across frames. This chain of thoughts should follow the output format: 
"Chain of Thoughts: 
- Frame 1: <analysis of frame 1>;
- Frame 2: <analysis of frame 2>;
...". For the analysis of each frame, you also have to follow the chain-of-thought format: 
"- *<question 1>* <answer 1>;
- *<question 2>* <answer2>; 
...", where you have to ask questions to yourself and answer them. Your answers should be as detailed as possible. Start with broader questions, like "what can be seen in the frame?" and proceed to more detailed questions such as "are there any other objects that haven't been listed?", "which objects could plausibly match the user query?", and "if more than one could match, which single instance is the best match and why?". There will be many questions and answers in the analysis of each frame; the exact questions vary by case. Generate in-depth questions and answers from your previous analysis. Your thinking must converge on one keyframe for the one best-matching object. When choosing the keyframe, prefer frames where the chosen object is not heavily overlapped by other objects, to help recognition. Partial visibility is acceptable if that frame is still the best choice.

Finally, output exactly one result entry using this format: "Output list: [{{object_index: 1, keyframe: k, object_description: <description of the chosen object in keyframe k>}}]". The list must contain at most one dictionary. Always use object_index: 1. k is the k-th keyframe in the grid. object_description should locate the object in that frame and help a downstream model find it. For example: "Output list: [{{object_index: 1, keyframe: 4, object_description: "the man at the top left corner of the image"}}]". If no keyframe shows any reasonable match to the query, output "Output list: []". Do not add extra dictionaries for alternate matches. Keep the output list in text format, not JSON. The output list begins with the prefix "Output list: " on the same line as the opening bracket, as in "Output list: [...]". Do not start with a new line. \n

Here is a grid image with {num_keyframes} keyframes. The user query is "{query}". Follow the instruction and output the best keyframe for the single best-matching object.
"""

def parse_gpt_output(text):
    list_outputs = text.split("Output list: ")[-1]
    # Prepare the input string for parsing
    text_input = list_outputs.replace('object_index', '"object_index"').replace('keyframe', '"keyframe"').replace('object_description', '"object_description"')

    # Convert the string to a list of dictionaries
    output = ast.literal_eval(text_input)
    return output