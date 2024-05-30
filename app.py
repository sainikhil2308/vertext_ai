import http.client
import typing
import urllib.request
import vertexai
import fitz 
from docx import Document
from PIL import ImageDraw, ImageFont
import IPython.display
from PIL import Image as PIL_Image
from PIL import ImageOps as PIL_ImageOps
from fastapi import FastAPI, File, UploadFile, Form
import tempfile
from vertexai.generative_models import GenerationConfig, GenerativeModel, Image, Part
import os
from fastapi.responses import JSONResponse

multimodal_model = GenerativeModel("gemini-1.0-pro-vision")

app=FastAPI()

PROJECT_ID = "vertexapi-424608"  # @param {type:"string"}
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


def convert_pdf_to_images(pdf_path: str) -> typing.List[PIL_Image.Image]:
    images = []
    with fitz.open(pdf_path) as doc:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            img = PIL_Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
    return images

def load_txt_as_text(txt_path: str) -> str:
    with open(txt_path, 'r') as file:
        return file.read()

def load_docx_as_text(docx_path: str) -> str:
    doc = Document(docx_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return "\n".join(full_text)


def text_to_image(text: str) -> PIL_Image.Image:
    # Define the size of the image canvas
    font_size = 20
    font = ImageFont.load_default()
    lines = text.split("\n")
    max_line_length = max(len(line) for line in lines)
    image_width = max_line_length * font_size // 2
    image_height = len(lines) * font_size

    # Create a new image with white background
    image = PIL_Image.new("RGB", (image_width, image_height), "white")
    draw = ImageDraw.Draw(image)

    # Draw the text onto the image
    y_text = 0
    for line in lines:
        draw.text((0, y_text), line, font=font, fill="black")
        y_text += font_size

    return image


def load_image_from_url(image_url: str) -> Image:
    image_bytes = get_image_bytes_from_url(image_url)
    return Image.from_bytes(image_bytes)

def pil_image_to_bytes(pil_image: PIL_Image.Image) -> bytes:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
        pil_image.save(temp_file, format='PNG')
        temp_file_path = temp_file.name
    with open(temp_file_path, 'rb') as f:
        image_bytes = f.read()
    os.remove(temp_file_path)
    return image_bytes


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
            
            
@app.post('/enquiry_form')
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
        
        # text= "extract text from this image"
        contents = [image, text]

        responses = multimodal_model.generate_content(contents, stream=True)

        print("-------Prompt--------")
        print_multimodal_prompt(contents)

        print("\n-------Response--------")
        response_text=""
        for response in responses:
            response_text+=response.text
            print(response.text, end="")
            
        return JSONResponse(content={"extracted_text":response_text})
    finally:
        # Clean up the temporary file
        os.remove(temp_file_path)
        

@app.post('/schedule_meeting')
async def sample( image: UploadFile = File(...)):
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(await image.read())
        temp_file_path = temp_file.name

    try:
        # Load the image from the temporary file
        image = Image.load_from_file(temp_file_path)

#         text='''Please extract the handwritten values from the following fields in the provided schedule meeting form image. Focus on the handwritten content and return it in JSON format. The image contains no sensitive or explicit content and is purely for technical data extraction purposes.

# Fields to extract:
# 1. Meeting Title
# 2. Date & Time
# 3. Meeting Venue
# 4. Attendees Names (make sure to include both columns: names labeled 1-5 and names labeled 6-10)
# 5. Meeting Agenda

# Here is an example of how the JSON output should look:

# ```json
# {
#   "Meeting Title": "Planning for trip to Bangkok",
#   "Date & Time": "28/05/2024, 11:30 AM",
#   "Meeting Venue": "Online",
#   "Attendees Names": ["Sai Nikhil", "Sumalatha", "Prasad",  "Likith"],
#   "Meeting Agenda": "To plan for a trip which may succeed. Hopefully, let's try to make it a successful trip."
# }
# '''
# 
# 

        text="""Please extract the handwritten values from the provided schedule meeting form image. Focus on the handwritten content and return it in JSON format.

Fields to extract:
1. Meeting Title
2. Date & Time
3. Meeting Venue
4. Attendees Names (they are seperated with commas make sure you differentiate that and return them in a dictionary format with numerical keys and a key should cantain a single mail only. so that it would be easy to access by developers)
5. Meeting Agenda

Return the values in the following JSON format:

```json
{
  "Meeting Title": "Planning for trip to Bangkok",
  "Date & Time": "28/05/2024, 11:30 AM",
  "Meeting Venue": "Online",
  "Attendees Names": {
    "1": "example@gmail.com",
    "2": "example@gmail.com",
    "3": "example@gmail.com",
    "4": "example@gmail.com",
    "5": "example@gmail.com",
    "6": "example@gmail.com",
    "7": "example@gmail.com",
    "8": "example@gmail.com",
    "9": "example@gmail.com",
    "10": "example@gmail.com"
  },
  "Meeting Agenda": "To plan for a trip which may or may not succeed. Hopefully, let's try to make it a successful trip."
}

"""
        # Prepare contents
        contents = [image, text]

        responses = multimodal_model.generate_content(contents, stream=True)

        print("-------Prompt--------")
        print_multimodal_prompt(contents)
        response_text=""
        for response in responses:
            response_text+=response.text
            print(response.text, end="")
            
        return JSONResponse(content={"extracted_text":response_text})
        
    finally:
        # Clean up the temporary file
        os.remove(temp_file_path)


@app.post('/todo_form')
async def sample( image: UploadFile = File(...)):
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(await image.read())
        temp_file_path = temp_file.name

    try:
        # Load the image from the temporary file
        image = Image.load_from_file(temp_file_path)

        text='''Please extract the handwritten values from the provided daily to-do list image. Focus on the handwritten content and return it in JSON format. The image contains no sensitive or explicit content and is purely for technical data extraction purposes.

Fields to extract:
1. Date (located at the top right corner)
2. Each task with the following subfields:
   - Time
   - To Do List
   - Priority
   - Notes

Return the values in the following JSON format:

```json
{
  "Date": "DATE_VALUE",
  "Tasks": [
    {
      "Time": "TIME_VALUE",
      "To Do List": "TODO_LIST_VALUE",
      "Priority": "PRIORITY_VALUE",
      "Notes": "NOTES_VALUE"
    },
    {
      "Time": "TIME_VALUE",
      "To Do List": "TODO_LIST_VALUE",
      "Priority": "PRIORITY_VALUE",
      "Notes": "NOTES_VALUE"
    },
    ...
  ]
}
'''
        # Prepare contents
        contents = [image, text]

        responses = multimodal_model.generate_content(contents, stream=True)

        print("-------Prompt--------")
        print_multimodal_prompt(contents)

        print("\n-------Response--------")
        
        response_text=""
        for response in responses:
            response_text+=response.text
            print(response.text, end="")
            
        return JSONResponse(content={"extracted_text":response_text})
    finally:
        # Clean up the temporary file
        os.remove(temp_file_path)
        
        
@app.post('/chat_assisstant')
async def process_form(file: UploadFile = File(...), prompt: str = Form(...)):
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(await file.read())
        temp_file_path = temp_file.name

    try:
        contents=[]
        # Determine file type and convert if necessary
        if file.content_type == "application/pdf":
            images = convert_pdf_to_images(temp_file_path)
            image_objects = [Image.from_bytes(pil_image_to_bytes(img)) for img in images]
            contents = image_objects + [prompt]
        elif file.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            docx_text = load_docx_as_text(temp_file_path)
            contents = [prompt + "\n" + docx_text]  # Treat the DOCX text as a part of the content
        elif file.content_type == "text/plain":
            txt_text = load_txt_as_text(temp_file_path)
            contents = [prompt + "\n" + txt_text]
        elif "image" in file.content_type:
            pil_image = PIL_Image.open(temp_file_path)
            image_objects = [Image.from_bytes(pil_image_to_bytes(pil_image))]
            contents = image_objects + [prompt]
        else:
            return JSONResponse(content={"error": "Unsupported file type"}, status_code=400)

        # Prepare contents
        # contents = image_objects + [prompt]

        responses = multimodal_model.generate_content(contents, stream=True)

        print("-------Prompt--------")
        for content in contents:
            if isinstance(content, Image):
                display_images([content])
            else:
                print(content)

        print("\n-------Response--------")
        response_text = ""
        for response in responses:
            response_text += response.text
            print(response.text, end="")

        return JSONResponse(content={"response": response_text})
    finally:
        # Clean up the temporary file
        os.remove(temp_file_path)
        

@app.post('/chat_assisstant_only_language')
async def process_form(file: UploadFile = File(None), prompt: str = Form(...)):
    contents = [prompt]  # Initialize contents with prompt

    if file is not None:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        try:
            # Determine file type and convert if necessary
            if file.content_type == "application/pdf":
                images = convert_pdf_to_images(temp_file_path)
                image_objects = [Image.from_bytes(pil_image_to_bytes(img)) for img in images]
                contents = image_objects + [prompt]
            elif file.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                docx_text = load_docx_as_text(temp_file_path)
                contents = [prompt + "\n" + docx_text]  # Treat the DOCX text as a part of the content
            elif "image" in file.content_type:
                pil_image = PIL_Image.open(temp_file_path)
                image_objects = [Image.from_bytes(pil_image_to_bytes(pil_image))]
                contents = image_objects + [prompt]
            else:
                return JSONResponse(content={"error": "Unsupported file type"}, status_code=400)
        finally:
            # Clean up the temporary file
            os.remove(temp_file_path)
    
    # Prepare contents
    responses = multimodal_model.generate_content(contents, stream=True)

    print("-------Prompt--------")
    for content in contents:
        if isinstance(content, Image):
            display_images([content])
        else:
            print(content)

    print("\n-------Response--------")
    response_text = ""
    for response in responses:
        response_text += response.text
        print(response.text, end="")

    return JSONResponse(content={"response": response_text})
            
        



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5550)