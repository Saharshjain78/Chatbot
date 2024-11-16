import os 
import streamlit as st
import pandas as pd
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from dotenv import load_dotenv, find_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.sequential import SimpleSequentialChain
from langchain.chains.sequential import SequentialChain

from langchain_experimental.agents.agent_toolkits.python.base import create_python_agent
from langchain_experimental.tools import PythonREPLTool
from langchain.agents.agent_types import AgentType
from langchain_community.utilities import WikipediaAPIWrapper  # Updated import

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# LLM

# LLM





st.title('AI Assistant for Data Science')
st.write('This is a simple AI assistant for data science tasks')

with st.sidebar:
    st.title("Upload Your Dataset")
    st.subheader("Get Started with Data Science")
    st.write("""
        Welcome to the AI Assistant for Data Science! 
        To begin your data science journey, please upload a CSV file. 
        Your dataset will serve as the foundation for our analysis and exploration. 
        Together, we will uncover insights and solve your business challenges using cutting-edge machine learning models. 
        Let's get started and have some fun with data!
    """)
    st.divider()
    st.caption("<p style='text-align:center'>made by Saharsh Jain</p>", unsafe_allow_html=True)
if 'clicked' not in st.session_state:
    st.session_state.clicked = {1:False}

def clicked(button):
    st.session_state.clicked[button] = True

st.button("Let's get started", on_click=clicked, args=[1])    
if st.session_state.clicked[1]:
    user_csv = st.file_uploader("Upload your file here", type="csv")
    if user_csv is not None:
        user_csv.seek(0)
        df = pd.read_csv(user_csv, low_memory=False)
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")  # Provide the model parameter
        llm.temperature = 0.3  # Set the temperature separately if supported


        def get_gemini_response(question):
            response = llm.invoke(question)
            content = response.content  # Access the content attribute
            return content
        @st.cache_data
        def steps_eda():    
            response_content = get_gemini_response("What are the steps for Exploratory Data Analysis in short points?")
            st.markdown(response_content)
            return steps_eda


        pandas_agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)
        # Ensure df is defined before using it
        question = "what is the meaning of the columns"
        column_meaning = pandas_agent.run(question)
        st.write(column_meaning)
        



        @st.cache_data
        def function_agent():
            st.write("**Data Overview**")
            st.write("The first rows of your dataset look like this:")
            st.write(df.head())
            st.write("**Data Cleaning**")
            columns_df = pandas_agent.run("What are the meaning of the columns?")
            st.write(columns_df)
            missing_values = pandas_agent.run("How many missing values does this dataframe have? Start the answer with 'There are'")
            st.write(missing_values)
            duplicates = pandas_agent.run("Are there any duplicate values and if so where?")
            st.write(duplicates)
            st.write("**Data Summarisation**")
            st.write(df.describe())
            correlation_analysis = pandas_agent.run("Calculate correlations between numerical variables to identify potential relationships.")
            st.write(correlation_analysis)
            outliers = pandas_agent.run("Identify outliers in the data that may be erroneous or that may have a significant impact on the analysis.")
            st.write(outliers)
            new_features = pandas_agent.run("What new features would be interesting to create?.")
            st.write(new_features)

            return
        
        @st.cache_data
        def function_question_variable():
            st.line_chart(df, y=[user_question_variable])
            summary_statistics = pandas_agent.run(f"What are the mean, median, mode, standard deviation, variance, range, quartiles, skewness and kurtosis of {user_question_variable}")
            st.write(summary_statistics)
            normality = pandas_agent.run(f"Check for normality or specific distribution shapes of {user_question_variable}")
            st.write(normality)
            outliers = pandas_agent.run(f"Assess the presence of outliers of {user_question_variable}")
            st.write(outliers)
            trends = pandas_agent.run(f"Analyse trends, seasonality, and cyclic patterns of {user_question_variable}")
            st.write(trends)
            missing_values = pandas_agent.run(f"Determine the extent of missing values of {user_question_variable}")
            st.write(missing_values)
            return

        @st.cache_data
        def function_question_dataframe():
            dataframe_info = pandas_agent.run(user_question_dataframe)
            st.write(dataframe_info)
            return

        st.header("Explainatory data analysis")
        st.write("Exploratory Data Analysis (EDA) is an approach to analyzing datasets to summarize their main characteristics, often with visual methods. EDA is used to understand the data, discover patterns, and identify relationships between variables. The goal of EDA is to gain insights and inform further analysis. Here are some common steps in EDA:")

        with st.sidebar:
            with st.expander('What are the steps of EDA'):
                    response_content = get_gemini_response("What are the steps for Exploratory Data Analysis in short points?")
                    st.markdown(response_content)


                    # Create a pandas agent


        function_agent()
        st.write("The assistant has completed the EDA process. You can now explore the results and continue with your data analysis.")

        user_question_variable = st.text_input("What are the thing that you are interested in asking?")
        if user_question_variable is not None and user_question_variable !="":
            function_question_variable()
            st.subheader('Further study')

        if user_question_variable:
            user_question_dataframe = st.text_input("Is there anything else you want to ask!")
            if user_question_variable is not None and user_question_variable not in ["","No","no"]:
                function_question_dataframe()
            if user_question_dataframe in ["no","No"]:
                st.write("Thank you for using the assistant")
                if user_question_dataframe:
                    st.divider()
                    st.header("Data Science Problem")
                    st.write("The assistant can help you with a data science problem. Please provide a brief description of the problem you would like to solve.")
                    prompt = st.text_input("Add ur prompt here")
                    data_problem_tenplate = PromptTemplate(
                        input_variables=['business_problem'],
                        template='Convert the following business problem into a data science problem {business_problem}.'

                    )

                    data_problem_chain = LLMChain(llm= llm, prompt = data_problem_tenplate, verbose = True)

                    
                    if prompt:
                        response = data_problem_chain.invoke.run(business_problem= prompt)

                        st.write(response)
                        
