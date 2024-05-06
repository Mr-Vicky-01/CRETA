import streamlit as st
import os
from streamlit_chat import message
from PyPDF2 import PdfReader
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
import time
from dotenv import load_dotenv

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

llm = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.7)


template = """You are a friendly chat assistant called "CRETA" having a conversation with a human and you are created by Pachaiappan an AI Specialist.
provided document:
{provided_docs}
previous_chat:
{chat_history}
Human: {human_input}
Chatbot:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input", "provided_docs"], template=template
)

llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
)


previous_response = ""
provided_docs = ""
def conversational_chat(query):
    global previous_response, provided_docs
    for i in st.session_state['history']:
        if i is not None:
            previous_response += f"Human: {i[0]}\n Chatbot: {i[1]}\n"
    print(previous_response)
    provided_docs = "".join(st.session_state["docs"])
    result = llm_chain.predict(chat_history=previous_response, human_input=query, provided_docs=provided_docs)
    st.session_state['history'].append((query, result))
    return result

st.title("🤖CRETA")
st.text("I am CRETA Your Friendly Assitant")

if 'history' not in st.session_state:
    st.session_state['history'] = []

    
if 'docs' not in st.session_state:
    st.session_state['docs'] = []
    
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_url_text(url_link):
    try:
        loader = WebBaseLoader(url_link)
        loader.requests_per_second = 1
        docs = loader.aload()
        extracted_text = ""
        for page in docs:
            extracted_text += page.page_content
        return extracted_text
    except Exception as e:
        print(f"Error fetching or processing URL: {e}")
        return ""

def response_streaming(text):
    for i in text:
        yield i
        time.sleep(0.01)

with st.sidebar:
    st.title("Add a file for CRETA memory:")
    uploaded_files = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
    uploaded_url = st.text_input("Paste the Documentation URL:")
    
    if st.button("Submit & Process"):
        if uploaded_files or uploaded_url:
            with st.spinner("Processing..."):
                if uploaded_files:
                    st.session_state["docs"] += get_pdf_text(uploaded_files)
                
                if uploaded_url:
                    st.session_state["docs"] += get_url_text(uploaded_url)
                
                st.success("Processing complete!")
        else:
            st.error("Please upload at least one PDF file or provide a URL.")
            
            
if 'messages' not in st.session_state:
    st.session_state.messages = [{'role': 'assistant', "content": "I'm Here to help your questions"}]
    
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.write(message['content'])
        
user_input = st.chat_input("Ask Your Questions 👉..")
if user_input:
    st.session_state.messages.append({'role': 'user', "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)
        
    with st.spinner("Thinking..."):
        response = conversational_chat(user_input)

    with st.chat_message("assistant"):
        full_response = st.write_stream(response_streaming(response))
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)