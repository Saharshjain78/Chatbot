import os 
from apikey import apikey
import streamlit as st
import pandas as pd
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from dotenv import load_dotenv, find_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# LLM
llm = ChatGoogleGenerativeAI(model="gemini-pro")  # Provide the model parameter
llm.temperature = 0.3  # Set the temperature separately if supported

st.title('AI Assistant for Data Science')
st.write('This is a simple AI assistant for data science tasks')

with st.sidebar:
    st.title("Upload Your Dataset")
    st.subheader("Get Started with Data Science")
    st.write("""
        Welcome to the AI Assistant for Data Science! 
        To begin your data science journey, please upload a CSV file.
    """)
    st.divider()
    st.caption("<p style='text-align:center'>made by Saharsh Jain</p>", unsafe_allow_html=True)

def get_gemini_response(question):
    response = llm.invoke(question)
    content = response.content  # Access the content attribute
    return content

if 'clicked' not in st.session_state:
    st.session_state.clicked = {1: False}

def clicked(button):
    st.session_state.clicked[button] = True

st.button("Let's get started", on_click=clicked, args=[1])

if st.session_state.clicked[1]:
    st.header('Exploratory Data Analysis Part')
    st.subheader('Solution')
    question = "What are the steps for Exploratory Data Analysis?"
    response_content = get_gemini_response(question)
    st.markdown(response_content)  # Display the response content in a readable format