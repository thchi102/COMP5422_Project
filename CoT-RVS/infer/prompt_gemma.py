from utils.util import preprocess_prompt
import torch
from pathlib import Path

def prompt_gemma(model, processor, image_path, query, num_keyframes, max_new_tokens=512):
    
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": preprocess_prompt(query, num_keyframes)}
            ]
        }
    ]

    model_param = next(model.parameters())
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model_param.device, dtype=model_param.dtype)

    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        generation = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        generation = generation[0][input_len:]
    response = processor.decode(generation, skip_special_tokens=True)
    return response

def save_answer(
    query,
    answer,
    path,
    response_label="Gemma3 response",
):
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)

    with open(path, "w") as f:
        f.write(f"")
    with open(path, "a") as f:
        f.write(f"Query: {query}\n")
        f.write(f"{response_label}: {answer}\n")