# Chat with PDF using Gemini AI

This project is a Streamlit application that allows users to interact with PDF documents using the Gemini AI model. The application enables users to upload PDF files, extract text from them, split the text into chunks, and store these chunks in a FAISS vector store for efficient similarity search. Users can then ask questions about the content of the PDFs, and the application will use the Gemini AI model to generate responses based on the extracted text. If the answer is not found in the provided context, the application will search the internet for an answer using the Gemini model.

## Features

- Upload PDF files and extract text from them.
- Split extracted text into manageable chunks.
- Store text chunks in a FAISS vector store for efficient similarity search.
- Ask questions about the content of the PDFs and get responses from the Gemini AI model.
- Search the internet for answers if the answer is not found in the provided context.

## Installation

Follow these steps to set up and run the project:

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Clone the Repository

```sh
git clone https://github.com/your-repo/chat-with-pdf.git
cd chat-with-pdf
```markdown
# Installation Guide

This guide will help you set up and run the project on your local machine.

---

## Prerequisites

Ensure you have the following installed on your system:

- **Python** 3.8 or higher  
- **pip** (Python package installer)

---

## Clone the Repository

Clone the repository to your local machine using the following command:

```bash
git clone <repository_url>
cd <repository_name>
```

---

## Set Up a Virtual Environment

It is recommended to use a virtual environment to manage dependencies. Run the following commands to create and activate a virtual environment:

### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### On macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Install Dependencies

Install the required Python packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

---

## Set Up Environment Variables

Create a `.env` file in the root directory of the project and add your Google API key as follows:

```
GOOGLE_API_KEY=<your_google_api_key>
```

Replace `<your_google_api_key>` with your actual Google API key.

---

## Run the Application

You can run the Streamlit application using the following command:

```bash
streamlit run app.py
```

This will start the application, and you can access it in your web browser at [http://localhost:8501](http://localhost:8501).

---

## Additional Information

To run other scripts like `chatwithpdf.py`, use the following command:

```bash
python chatwithpdf.py
```

Make sure to upload the necessary PDF files or datasets as required by the application.

---

## Troubleshooting

If you encounter any issues:

1. Ensure that all dependencies are installed correctly.
2. Verify that your environment variables are set up properly.
3. Check the documentation of the respective libraries for additional help.

