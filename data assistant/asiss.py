import os
import streamlit as st
import pandas as pd
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
import time

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
llm.temperature = 0.3


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
    retries = 5
    delay = 2
    for i in range(retries):
        try:
            response = llm.invoke(question)
            return response.content
        except (genai.exceptions.ResourceExhausted, genai.exceptions.InternalServerError) as e:
            if i < retries - 1:
                time.sleep(delay)
                delay = min(delay * 2, 60)  # Exponential backoff with a cap at 60 seconds
            else:
                st.error(f"Failed to get response from API: {e}")
                return None

@st.cache_data
def load_csv(file):
    file.seek(0)
    return pd.read_csv(file, low_memory=False)

@st.cache_data
def steps_eda():
    response_content = get_gemini_response("What are the steps for Exploratory Data Analysis in short points?")
    return response_content

@st.cache_data
def get_column_meaning(_agent, question):
    return _agent.run(question)

if 'clicked' not in st.session_state:
    st.session_state.clicked = {1: False}

def clicked(button):
    st.session_state.clicked[button] = True

st.button("Let's get started", on_click=clicked, args=[1])

if st.session_state.clicked[1]:
    user_csv = st.file_uploader("Upload your file here", type="csv")
    if user_csv is not None:
        df = load_csv(user_csv)
        pandas_agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)
        
        st.header('Exploratory Data Analysis Part')
        st.subheader('Solution')
        st.markdown(steps_eda())
        
        question = "what is the meaning of the columns"
        column_meaning = get_column_meaning(pandas_agent, question)
        st.write(column_meaning)

        # Combine multiple operations into fewer requests
        combined_analysis = pandas_agent.run("""
            How many missing values does this dataframe have? Start the answer with 'There are'.
            Are there any duplicate values if so where?
            Calculate correlations between numerical variables to identify potential relationships.
            Identify outliers in the data that may be erroneous or that may have a significant impact on the analysis.
            What new features would be interesting to create?
        """)
        st.write(combined_analysis)

        def function_question_variable(user_question_variable):
            try:
                st.line_chart(df, y=[user_question_variable])
                summary_statistics = pandas_agent.run(
                    question=f"""
                        What are the mean, median, mode, standard deviation, variance, range, quartiles, skewness and kurtosis of {user_question_variable}?
                        Check for normality or specific distribution shapes of {user_question_variable}.
                        Assess the presence of outliers of {user_question_variable}.
                        Analyse trends, seasonality, and cyclic patterns of {user_question_variable}.
                        Determine the extent of missing values of {user_question_variable}.
                    """,
                    handle_parsing_errors=True
                )
                st.write(summary_statistics)
            except Exception as e:
                st.error(f"An error occurred: {e}")
                if st.button("Retry"):
                    summary_statistics = pandas_agent.run(
                        question=f"""
                            What are the mean, median, mode, standard deviation, variance, range, quartiles, skewness and kurtosis of {user_question_variable}?
                            Check for normality or specific distribution shapes of {user_question_variable}.
                            Assess the presence of outliers of {user_question_variable}.
                            Analyse trends, seasonality, and cyclic patterns of {user_question_variable}.
                            Determine the extent of missing values of {user_question_variable}.
                        """,
                        handle_parsing_errors=True
                    )
                    st.write(summary_statistics)

        def function_question_dataframe(user_question_dataframe):
            dataframe_info = pandas_agent.run(user_question_dataframe)
            st.write(dataframe_info)
            return
        
        st.header("Explainatory data analysis")
        st.write("Exploratory Data Analysis (EDA) is an approach to analyzing datasets to summarize their main characteristics, often with visual methods. EDA is used to understand the data, discover patterns, and identify relationships between variables. The goal of EDA is to gain insights and inform further analysis. Here are some common steps in EDA:")

        with st.sidebar:
            with st.expander('What are the steps of EDA'):
                response_content = get_gemini_response("What are the steps for Exploratory Data Analysis in short points?")
                st.markdown(response_content)

        st.write("The assistant has completed the EDA process. You can now explore the results and continue with your data analysis.")

        user_question_variable = st.selectbox("Select a column to analyze", df.columns)

        if user_question_variable:
            function_question_variable(user_question_variable)

        user_question_dataframe = st.text_input("Is there anything else you want to ask?")
        
        if user_question_dataframe and user_question_dataframe.lower() not in ["", "no"]:
            function_question_dataframe(user_question_dataframe)
        
        if user_question_dataframe.lower() == "no":
            st.write("Thank you for using the assistant")
            
            st.divider()
            st.header("Data Science Problem")
            st.write("The assistant can help you with a data science problem. Please provide a brief description of the problem you would like to solve.")
            
            prompt = st.text_input("Add your prompt here")
            
            data_problem_template = PromptTemplate(
                input_variables=['business_problem'],
                template='Convert the following business problem into a data science problem: {business_problem}.'
            )

            data_problem_chain = LLMChain(llm=llm, prompt=data_problem_template, verbose=True)

            if prompt:
                try:
                    response = data_problem_chain.invoke({'business_problem': prompt})
                    st.write(response)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    if st.button("Retry"):
                        response = data_problem_chain.invoke({'business_problem': prompt})
                        st.write(response)