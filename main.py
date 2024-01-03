from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from PIL import Image
import requests
from io import BytesIO

import services.image_captioning as image_captioning

app = FastAPI()

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    try:
        # Open the uploaded image and convert it to RGB
        img = Image.open(BytesIO(await file.read())).convert('RGB')

        # Generate caption for the image
        caption = image_captioning.generate_caption(img)

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
