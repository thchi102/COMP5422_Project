from openai import AzureOpenAI
import base64
from mimetypes import guess_type
from utils.util import preprocess_prompt
def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"

def prompt_openai(client, model, data_url, query, num_keyframes, max_tokens=2500):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role":"system","content":"You are a helpful assistant that answers question in chain of thoughts."},
            {"role":"user","content":[  
                { 
                    "type": "text", 
                    "text": preprocess_prompt(query, num_keyframes)
                },
                { 
                    "type": "image_url",
                    "image_url": {
                        "url": data_url
                    }
                }
            ]
            },
        ],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content
def prompt_openai_without_cot(client, model,data_url,query,num_keyframes):
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role":"system","content":"You are a helpful assistant that answers question in chain of thoughts."},
            {"role":"user","content":[  
                { 
                    "type": "text", 
                    "text": f"""You will act as a keyframe selection agent for a video reasoning task. During each inference, you will be given a grid image that contains multiple keyframes sampled from a long video. The keyframes are aligned from top to down or from left to right, following their temporal order. You will also be given a user query. You must identify exactly ONE target: the single object or instance that best matches the query. If several candidates could match, pick the single best one (most specific, most relevant, best visibility, least overlap with other objects). You need to find the best keyframe for that one object, where a segmentation model can find it with minimal effort. You have to output a list with at most one dictionary, using the format: "Output list: [{{object_index: 1, keyframe: k, object_description: <description of the chosen object in keyframe k>}}]". Always use object_index: 1. k is the k-th keyframe in the grid. object_description must include the location of the object in that frame. For example: "Output list: [{{object_index: 1, keyframe: 4, object_description: "the man at the top left corner of the image"}}]". If no reasonable match exists, use "Output list: []". Do not list multiple objects. Partial visibility of the chosen object is acceptable. Prefer keyframes where the object is not heavily overlapped. Keep the output list in text format, not JSON. The line must be "Output list: [...]" with no leading newline. Do not include anything after the output list.
Here is a grid image with {num_keyframes} keyframes. The user query is "{query}". Follow the instruction and output the best keyframe for the single best-matching object.
"""
                },
                { 
                    "type": "image_url",
                    "image_url": {
                        "url": data_url
                    }
                }
            ]
            },
        ],
        max_tokens=2500
    )
    return response.choices[0].message.content
