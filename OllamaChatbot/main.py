import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv
load_dotenv()

# LANGSMITH TRACKING
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot with Ollama"

# PROMPT TEMPLATE
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries."),
        ("user", "Question:{question}")
    ]
)

def generate_response(question, llm, temperature, max_tokens):
    llm = Ollama(
        model=llm, 
        temperature=temperature, 
        num_predict=max_tokens
    )
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({"question": question})
    return answer

# streamlit App configuration
st.set_page_config(
    page_title="Ollama Q&A Chatbot", 
    page_icon="🤖",
    #layout="wide"
)
# streamlit app
st.title("🚀 Enhanced Q&A Chatbot With Ollama")
st.sidebar.title("⚙️ Model Settings")

# drop down to select models
llm = st.sidebar.selectbox("Select an Opensource model", ["llama3", "gemma2:2b", "gemma2", "phi3"])

# adjust response parameters
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

# sidebar Tips
st.sidebar.markdown("### 💡 Tips")
st.sidebar.info(
    "- Choose smaller models for faster responses\n"
    "- Lower temperature for more focused answers\n"
    "- Larger models provide more detailed responses"
)

# main interface for user input
with st.form(key='chat_form'):
    st.write("Ask any question")
    user_input = st.text_input("You:")
    submit_button = st.form_submit_button("Send")

# response generation
if submit_button and user_input:
    with st.spinner('Generating response...'):
        response = generate_response(user_input, llm, temperature, max_tokens)
    st.write(response)
else:
    st.write("Please provide the query")