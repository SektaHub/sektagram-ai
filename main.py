from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from PIL import Image
import requests
from io import BytesIO

import services.image_captioning as image_captioning
import services.sentence_embeddings as sentence_embeddings

app = FastAPI()

@app.post("/api/generateCaptionFromUpload/")
async def generate_caption_from_upload(file: UploadFile = File(...)):
    try:
        # Open the uploaded image and convert it to RGB
        img = Image.open(BytesIO(await file.read())).convert('RGB')

        # Generate caption for the image
        caption = image_captioning.generate_caption(img)

        return {"filename": file.filename, "caption": caption}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the image: {str(e)}")

@app.post("/api/generateCaptionFromLink/")
async def generate_caption_from_link(image_link: str):
    try:
        # Download the image from the link and convert it to RGB
        img = Image.open(requests.get(image_link, stream=True).raw).convert('RGB')

        # Generate caption for the image
        caption = image_captioning.generate_caption(img)

        return {"image_link": image_link, "caption": caption}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the image from the link: {str(e)}")


@app.post("/api/embed_sentence")
async def embed_sentence(sentence: str):
    embedding = sentence_embeddings.embed_sentence(sentence)
    return {"embedding": embedding}


@app.get("/")
async def main():
    content = """
    <body>
    <form action="/api/generateCaptionFromUpload/" enctype="multipart/form-data" method="post">
    <input type="file" name="file" accept="image/*">
    <input type="submit" value="Upload Image">
    </form>
    </body>
    """
    return HTMLResponse(content=content)


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
