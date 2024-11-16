import os
import pandas as pd
import PyPDF2
import google.generativeai as genai
import sqlalchemy
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import streamlit as st
import spacy
from sqlalchemy import text
import sqlparse


# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Database connection setup (example with SQLite)
DATABASE_URL = "sqlite:///your_database.db"
engine = sqlalchemy.create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)


def is_valid_sql(query):
    try:
        # Parse the SQL query
        parsed = sqlparse.parse(query)
        # Check if it's a SELECT statement
        if parsed and parsed[0].get_type() == 'SELECT':
            return True
    except:
        pass
    return False

def read_excel(file):
    # Read Excel file into a pandas DataFrame
    return pd.read_excel(file)

def read_pdf(file):
    # Read PDF file and extract text
    reader = PyPDF2.PdfFileReader(file)
    text = ""
    for page in range(reader.numPages):
        text += reader.getPage(page).extractText()
    return text

def get_gemini_response(question, context=""):
    # Generate a response using the Gemini model
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content([context, question])
    return response.text

def generate_query(prompt, context=""):
    gemini_response = get_gemini_response(f"Generate a valid SQL query for SQLite based on this prompt based on the date provided in the database: {prompt}", context)
    if is_valid_sql(gemini_response):
        return gemini_response
    return None

def execute_query(query):
    try:
        with Session() as session:
            result = session.execute(text(query))
            return result.fetchall()
    except sqlalchemy.exc.OperationalError as e:
        st.error(f"An error occurred while executing the query: {e}")
        return None

import re

def sanitize_input(input_str):
    # Basic sanitization to remove potentially harmful characters or patterns
    return re.sub(r"[^\w\s]", "", input_str)

def app():
    st.title("Database Chatbot")
    
    # File uploader for Excel and PDF
    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx", "pdf"])
    file_content = ""
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            file_content = read_pdf(uploaded_file)
        else:
            df = read_excel(uploaded_file)
            file_content = df.to_json()  # Convert DataFrame to JSON for processing

    user_prompt = st.text_input("Enter your query:")
    if user_prompt:
        sanitized_prompt = sanitize_input(user_prompt)
        query = generate_query(sanitized_prompt, file_content)
        if query:
            st.text(f"Generated Query: {query}")
            try:
                results = execute_query(query)
                if results:
                    st.write(results)
                else:
                    st.write("No results found.")
            except sqlalchemy.exc.OperationalError as e:
                st.error(f"SQL syntax error: {e}")
            except Exception as e:
                st.error(f"An error occurred while executing the query: {e}")
        else:
            st.error("Failed to generate a valid SQL query. Please try rephrasing your question.")

if __name__ == "__main__":
    app()