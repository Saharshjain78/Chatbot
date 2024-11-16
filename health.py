from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import os
import google.generativeai as genai
from PIL import Image

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(input, image, prompt):
    model = genai.GenerativeModel('gemini-1.5-pro')
    response = model.generate_content([input, image[0], prompt])
    return response.text

def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        byte_data = uploaded_file.getvalue()
        image_parts = [
            {
                "mime_type": uploaded_file.type,
                "data": byte_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")

# Initialize our streamlit app
st.set_page_config(page_title="Gemini Health App")
st.header("Gemini Health App")
input = st.text_input("Input Prompt: ", key="input")
uploaded_file = st.file_uploader("Choose uploaded file", type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

submit = st.button("Tell me the amount of total calories")

input_prompt = """
You are an expert Nutritionist and mathematician where you need to calculate the size of the utensils and see and identify the food items from the image and calculate the total calories, also provide the details of every food item with calorie intake
in the below format-
1. Item one: no of calories
2. Item two: no of calories
.....
.....
.....
"""

if submit:
    image_data = input_image_setup(uploaded_file)
    response = get_gemini_response(input_prompt, image_data, input)
    st.subheader("Response")
    st.write(response)