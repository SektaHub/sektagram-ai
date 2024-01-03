from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from PIL import Image
import requests
from io import BytesIO

from transformers import AutoProcessor, Blip2ForConditionalGeneration, BlipProcessor, BlipForConditionalGeneration
import torch

app = FastAPI()

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


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    try:
        # Open the uploaded image and convert it to RGB
        img = Image.open(BytesIO(await file.read())).convert('RGB')

        # Generate caption for the image
        caption = generate_caption(img)

        return {"filename": file.filename, "caption": caption}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the image: {str(e)}")

@app.get("/")
async def main():
    content = """
    <body>
    <form action="/uploadfile/" enctype="multipart/form-data" method="post">
    <input type="file" name="file" accept="image/*">
    <input type="submit" value="Upload Image">
    </form>
    </body>
    """
    return HTMLResponse(content=content)


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
