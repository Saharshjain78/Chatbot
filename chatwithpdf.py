import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import re  # Import the 're' module for regular expressions

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""  # Ensure text is not None
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def clean_text(text):
    # Replace problematic characters (e.g., surrogates) with generic alternatives
    text = re.sub(r"[\ud800-\udfff]", " ", text) 
    return text

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context, search the answer on the internet and say "answer is not available in the context" along with the answer from the internet, don't provide the wrong answer\n\n
    Context:\n{context}?\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def search_internet_with_gemini(question):
    gemini_model = genai.GenerativeModel("gemini-pro")
    gemini_response = gemini_model.generate_content(question)
    return gemini_response.text  # Return the text of the response

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    if response["output_text"].strip().lower() == "answer is not available in the context":
        internet_response = search_internet_with_gemini(user_question)
        return {"output_text": f"The answer was not found in the provided context. Here's information from a general search:\n\n{internet_response}"}
    
    return response

def main():
    st.set_page_config(page_title="Chat PDF")
    st.header("Chat with PDF using Gemini💁")
    
    if "chat_history" not in st.session_state:
        st.session_state['chat_history'] = []

    user_question = st.text_input("Ask a Question from the PDF Files", key="input")
    submit = st.button("Ask the question")

    if submit and user_question:
        st.session_state['chat_history'].append(("You", user_question))
        st.subheader("The Response is")
        response = user_input(user_question)
        
        st.write("Reply:", response["output_text"])
        st.session_state["chat_history"].append(("Bot", response["output_text"]))

    st.subheader("The chat history is")
    for role, text in st.session_state["chat_history"]:
        st.write(f"{role}: {text}")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                raw_text = clean_text(raw_text)  # Clean the text before splitting
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()