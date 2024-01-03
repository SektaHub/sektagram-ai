
from transformers import AutoProcessor, Blip2ForConditionalGeneration, BlipProcessor, BlipForConditionalGeneration
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
has_gpu = torch.cuda.is_available()
print("GPU avaliable: " + str(has_gpu))

#model = "Salesforce/blip2-opt-2.7b"
model = "Salesforce/blip-image-captioning-large"

# Load the Blip2 model with int8 quantization
processor = BlipProcessor.from_pretrained(model)
#model = BlipForConditionalGeneration.from_pretrained(model, device_map=device, load_in_8bit=has_gpu)
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float16).to(device)



def generate_caption(image):
    inputs = processor(image, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(**inputs, max_new_tokens=20)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return generated_text

