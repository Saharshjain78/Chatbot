import os
import pandas as pd
import google.generativeai as genai
import sqlalchemy
from dotenv import load_dotenv
import streamlit as st
import sqlite3

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_response(question, prompt):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content([prompt[0], question])
    return response.text

def read_sql_query(sql, db):
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    conn.commit()
    conn.close()
    return rows

prompt = [
    """
You are an elite SQL query generator, specializing in translating natural language requests into precise SQL code. Your task is to:

Carefully analyze the user's input to identify:
Required tables
Specific columns
Desired operations (e.g., SELECT, JOIN, GROUP BY, etc.)
Any conditions or filters

Construct a SQL query that:
Uses only the explicitly mentioned or clearly implied tables and columns
Incorporates user-friendly aliases for readability
Follows best practices for SQL syntax and structure
Achieves the exact outcome requested, without superfluous operations

Present the SQL code directly, without:
Surrounding code blocks ( Also the sql code should not have ''' in the beginning or end and sql word on the o/p.)
Explanatory text, unless the user specifically requests it
Assumptions or additional information beyond the scope of the request

If the input lacks critical details:
Briefly note the missing information
Provide a partial query where possible, using placeholders for unknown elements

Prioritize:
Accuracy in translating the request
Efficiency in query design
Clarity and readability of the generated SQL

Remember: Your role is to generate SQL code, not to explain or justify it. Deliver clean, functional queries that directly address the user's needs.
    """
]

st.set_page_config(page_title="Data Analysis Bot!")
st.title("Data Analysis Bot!")

# File uploader
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Read the file
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    
    # Extract the column names (headings)
    headings = df.columns.tolist()
    
    # Display the headings
    st.subheader("Extracted Headings:")
    st.write(headings)
    
    # Store the data in SQLite database
    engine = sqlalchemy.create_engine("sqlite:///data.db")
    df.to_sql("data_table", engine, if_exists="replace", index=False)
    
    st.success("Data uploaded and stored in the database!")
    
    # Display a preview of the data
    st.subheader("Data Preview:")
    st.dataframe(df.head())
    
    # Input for the user's question
    question = st.text_input("Ask a question about the data:")
    
    if st.button("Generate Query and Run"):
        if question:
            # Generate SQL query
            generated_query = get_response(question, prompt)
            
            st.subheader("Generated SQL Query:")
            st.code(generated_query)
            
            try:
                # Execute the query
                result = read_sql_query(generated_query, "data.db")
                
                st.subheader("Query Result:")
                if result:
                    # Create DataFrame from result, inferring column names
                    result_df = pd.DataFrame(result)
                    if len(result_df.columns) == 1:
                        # If only one column, name it based on the query
                        result_df.columns = ['Result']
                    st.dataframe(result_df)
                else:
                    st.info("The query returned no results.")
            except Exception as e:
                st.error(f"Error executing the query: {str(e)}")
        else:
            st.warning("Please enter a question to generate a query.")
else:
    st.info("Please upload a CSV or Excel file to begin.")