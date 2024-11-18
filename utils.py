import getpass
import re 
import warnings
import json
import os
import streamlit as st
import textwrap
from langchain_community.document_loaders import CSVLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# Set configuration
load_dotenv()
warnings.filterwarnings("ignore")

# Hydration recommendation
def get_hydration_recommendation(daily_water_intake):
    # General hydration recommendation
    recommended_intake = 2.5  # liters
    if daily_water_intake < recommended_intake:
        return f"Your intake is below the recommended level. Try to increase your daily intake to at least {recommended_intake} liters."
    else:
        return "Your hydration is on track!"

# Stress level management
def get_stress_management(stress_level):
    if stress_level <= 3:
        return "Stress is low. Keep up the good work!"
    elif 4 <= stress_level <= 7:
        return "Moderate stress. Try relaxation techniques such as deep breathing or meditation."
    else:
        return "High stress. Consider consulting a healthcare professional for stress management."

# Sleep improvement suggestions
def get_sleep_improvement(sleep_hours):
    if sleep_hours < 6:
        return "You're getting less than the recommended amount of sleep. Aim for 7-9 hours for optimal health."
    elif 6 <= sleep_hours <= 8:
        return "Your sleep duration is good. Keep it up!"
    else:
        return "You might be sleeping too much. Aim for 7-9 hours of sleep per night."

# Nutritional diet plan
def get_diet_plan(diet_preference):
    if diet_preference == "Vegetarian":
        return "Consider including a variety of fruits, vegetables, legumes, and whole grains in your diet."
    elif diet_preference == "Non-Vegetarian":
        return "Include lean meats, fish, eggs, and plant-based foods to balance your nutrition."
    else:
        return "For a vegan diet, focus on plant-based sources of protein such as beans, tofu, and lentils."

# Function for processing PDF text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Wrap text to preserve newlines
def wrap_text_preserve_newlines(text, width=110):
    lines = text.split('\n')
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text

# Function to get vector store
def get_vector_store(text_chunks, gemini_api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gemini_api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Conversational chain for QA
def get_conversational_chain(db, gemini_api_key):
    prompt_template = """
    Analyze the provided medical report based on the context and question given. Provide:
1. A detailed health analysis based on the medical report data.
2. Identification of abnormal values (if any) and their significance.
3. Recommendations for medicines or supplements (advising consultation with a doctor).
4. Suggestions for dietary adjustments and lifestyle changes to improve health.
5. Preventive measures to maintain or enhance overall well-being.

If all values are within normal ranges, provide general health advice based on the context. If any information is missing in the context or the report, explicitly state: "The answer is not available in the provided context."

*Context*: {context}  
*Question*: {question}  

*Health Report*:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperatur=0.3, google_api_key=gemini_api_key)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gemini_api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    db = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)
    chain = RetrievalQA.from_chain_type(llm=model, chain_type="stuff", retriever=db.as_retriever(), chain_type_kwargs={"prompt": prompt})
    return chain

# Function for user input handling
def user_input(user_question, gemini_api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gemini_api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain(new_db, gemini_api_key)
    
    response = chain.invoke(user_question)
    st.write(wrap_text_preserve_newlines(response['result']))