import http.client
import typing
import urllib.request
import vertexai
import IPython.display
from PIL import Image as PIL_Image
from PIL import ImageOps as PIL_ImageOps
from fastapi import FastAPI, File, UploadFile, Form
import tempfile
from vertexai.generative_models import GenerationConfig, GenerativeModel, Image, Part
import os

multimodal_model = GenerativeModel("gemini-1.0-pro-vision")

app=FastAPI()

PROJECT_ID = " vertexapi-424608"  # @param {type:"string"}
LOCATION = "asia-south1"  # @param {type:"string"}

vertexai.init(project=PROJECT_ID, location=LOCATION)

def display_images(
    images: typing.Iterable[Image],
    max_width: int = 600,
    max_height: int = 350,
) -> None:
    for image in images:
        pil_image = typing.cast(PIL_Image.Image, image._pil_image)
        if pil_image.mode != "RGB":
            # RGB is supported by all Jupyter environments (e.g. RGBA is not yet)
            pil_image = pil_image.convert("RGB")
        image_width, image_height = pil_image.size
        if max_width < image_width or max_height < image_height:
            # Resize to display a smaller notebook image
            pil_image = PIL_ImageOps.contain(pil_image, (max_width, max_height))
        IPython.display.display(pil_image)


def get_image_bytes_from_url(image_url: str) -> bytes:
    with urllib.request.urlopen(image_url) as response:
        response = typing.cast(http.client.HTTPResponse, response)
        image_bytes = response.read()
    return image_bytes


def load_image_from_url(image_url: str) -> Image:
    image_bytes = get_image_bytes_from_url(image_url)
    return Image.from_bytes(image_bytes)


def get_url_from_gcs(gcs_uri: str) -> str:
    # converts gcs uri to url for image display.
    url = "https://storage.googleapis.com/" + gcs_uri.replace("gs://", "").replace(
        " ", "%20"
    )
    return url


def print_multimodal_prompt(contents: list):
    """
    Given contents that would be sent to Gemini,
    output the full multimodal prompt for ease of readability.
    """
    for content in contents:
        if isinstance(content, Image):
            display_images([content])
        elif isinstance(content, Part):
            url = get_url_from_gcs(content.file_data.file_uri)
            IPython.display.display(load_image_from_url(url))
        else:
            print(content)
            
            
@app.post('/hello')
async def sample( image: UploadFile = File(...)):
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(await image.read())
        temp_file_path = temp_file.name

    try:
        # Load the image from the temporary file
        image = Image.load_from_file(temp_file_path)

        text='''Extract the handwritten values from the following fields in the student enquiry form:
            1. Date
            2. Time
            3. Name of the Candidate
            4. Father/Husband Name
            5. Mother's Name
            6. Aadhar No
            7. Contact Number
            8. Email Id
            9. Gender
            10. Date of Birth
            11. Education Level
            12. Contact Address
            13. Pin Code
            14. Purpose
            Return the values in JSON format.
            Here is an example of how the JSON output should look:
            ```json
            {
            "Date": "23/08/2024",
            "Time": "11:53 AM",
            "Name of the Candidate": "Example Name",
            "Father/Husband Name": "Example Father's Name",
            "Mother's Name": "Example Mother's Name",
            "Aadhar No": "1234 5678 9012",
            "Contact Number": "9876543210",
            "Email Id": "example@example.com",
            "Gender": "Female",
            "Date of Birth": "01/01/2000",
            "Education Level": "Degree",
            "Address": "Example Address",
            "Pin Code": "123456",
            "purpose": "Admisssion"
            }'''
        # Prepare contents
        contents = [image, text]

        responses = multimodal_model.generate_content(contents, stream=True)

        print("-------Prompt--------")
        print_multimodal_prompt(contents)

        print("\n-------Response--------")
        for response in responses:
            print(response.text, end="")
    finally:
        # Clean up the temporary file
        os.remove(temp_file_path)
        
        
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5555)