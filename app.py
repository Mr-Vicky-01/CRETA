import streamlit as st
import os
from streamlit_chat import message
from PyPDF2 import PdfReader
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from langchain import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import WebBaseLoader

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

llm = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.2)


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
            previous_response += f"Human: {i[0]}\n Chatbot: {i[1]}"
    docs = ""
    for j in st.session_state["docs"]:
        if j is not None:
            docs += j
    provided_docs = docs
    result = llm_chain.predict(chat_history=previous_response, human_input=query, provided_docs=provided_docs)
    st.session_state['history'].append((query, result))
    return result

st.title("CRETA 🤖")
st.text("I am CRETA Your Friendly Assitant")

if 'history' not in st.session_state:
    st.session_state['history'] = []
    
# Initialize messages
if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello ! Ask me anything"]

if 'past' not in st.session_state:
    st.session_state['past'] = [" "]
    
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
            
            
# Create containers for chat history and user input
response_container = st.container()
container = st.container()

# User input form
user_input = st.chat_input("Ask Your Questions 👉..")
with container:
    if user_input:
        output = conversational_chat(user_input)
        # answer = response_generator(output)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)
        
        
# Display chat history
if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            if i != 0:
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="adventurer")
            message(st.session_state["generated"][i], key=str(i), avatar_style="bottts")
            