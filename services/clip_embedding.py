from transformers import CLIPTokenizerFast, CLIPProcessor, CLIPModel
import torch
from PIL import Image
import torchvision.transforms as transforms
from typing import Any, Dict

# if you have CUDA or MPS, set it to the active device like this
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
model_id = "openai/clip-vit-large-patch14"

# we initialize a tokenizer, image processor, and the model itself
tokenizer = CLIPTokenizerFast.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)
model = CLIPModel.from_pretrained(model_id).to(device)


def embed_image(image: Image.Image) -> torch.Tensor:
    try:
        inputs = processor(images=image, return_tensors="pt", padding=True)
        inputs = inputs.to(device)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        return image_features
    except Exception as e:
        raise ValueError(f"Error processing the image: {str(e)}")


def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    inputs = inputs.to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    return text_features
