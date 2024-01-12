from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from PIL import Image
import requests
from io import BytesIO

import services.image_captioning as image_captioning
import services.sentence_embeddings as sentence_embeddings

app = FastAPI()

dotnet_backend_url = "http://localhost:8080/api"


class SentenceInput(BaseModel):
    sentence: str


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
async def embed_sentence(sentence_input: SentenceInput):
    try:
        sentence = sentence_input.sentence
        embedding = sentence_embeddings.embed_sentence(sentence)
        return {"embedding": embedding}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing sentence: {str(e)}")


# FastAPI endpoint to generate captions for images without captions from .NET backend
@app.post("/api/generateCaptionsForImagesWithoutCaption")
async def generate_captions_for_images_without_caption():
    try:
        # Get the list of images without captions from the .NET backend
        dotnet_endpoint_url = f"{dotnet_backend_url}/Image/GetImagesWithoutCaption"
        response = requests.get(dotnet_endpoint_url)
        image_list = response.json()

        # Iterate through the images and generate captions
        captions = []
        for image in image_list:
            image_id = image["id"]
            image_url = f"{dotnet_backend_url}/Image/{image_id}/Content"

            # Download the image from the .NET backend
            image_response = requests.get(image_url, stream=True)
            if image_response.status_code == 200:
                # Convert the image to RGB
                img = Image.open(BytesIO(image_response.content)).convert('RGB')

                # Generate caption for the image
                new_caption = image_captioning.generate_caption(img)

                # Generate CaptionEmbedding
                caption_embedding = sentence_embeddings.embed_sentence(new_caption)

                rounded_embedding = [round(value, 8) for value in caption_embedding]

                # Create JSON Patch request
                patch_request = [
                    {"op": "replace", "path": "/generatedCaption", "value": new_caption},
                ]

                # Send JSON Patch request to .NET backend
                patch_url = f"{dotnet_backend_url}/Image/{image_id}"
                patch_headers = {'Content-Type': 'application/json-patch+json'}  # Ensure correct content-type header is set for patch requests
                patch_response = requests.patch(patch_url, json=patch_request, headers=patch_headers)

                embedding_patch_request = {
                    "embedding": str(caption_embedding),
                }

                patch_url = f"{dotnet_backend_url}/Image/{image_id}/PatchCaptionEmbedding"
                patch_response2 = requests.patch(patch_url, json=embedding_patch_request, headers=patch_headers)

                print("REQ:" + str(embedding_patch_request))

                if patch_response.status_code == 204 and patch_response2.status_code == 204:
                    captions.append({"image_id": image_id, "caption": new_caption})
                else:
                    captions.append(
                        {"image_id": image_id, "error": f"Error updating caption: {patch_response.status_code}, {patch_response2.content}"})
            else:
                # Handle errors, if any
                captions.append(
                    {"image_id": image_id, "error": f"Error downloading image: {image_response.status_code}"})

        return {"captions": captions}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing images: {str(e)}")


@app.get("/")
async def main():
    content = """
    <body>
        <form action="/api/generateCaptionsForImagesWithoutCaption" method="post">
            <input type="submit" value="Generate Captions for Images Without Caption">
        </form>
        <br>
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
