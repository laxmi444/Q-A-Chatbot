import streamlit as st
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


import os
from dotenv import load_dotenv
load_dotenv()

# LANGSMITH TRACKING
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Q&A Chatbot with Google Gemini"
 

# PROMPT TEMPLATE
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please respond to the user queries."),
        ("user","Question:{question}")
    ]

)


def generate_response(question,api_key,llm,temperature,max_tokens): # passing api key during runtime
    genai.api_key=api_key
    llm=ChatGoogleGenerativeAI(model=llm)
    output_parser=StrOutputParser()
    chain=prompt|llm|output_parser
    answer=chain.invoke({"question":question})
    return answer

## streamlit app
st.title("Enchanced Q&A Chabot With Gemini")
st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter your Google Gemini API key:",type="password")

# drop down to select gemini models
llm=st.sidebar.selectbox("Select a Gemini model",["gemini-2.0-flash-exp","gemini-1.5-flash","gemini-1.5-flash-8b","gemini-1.5-pro"])

# adjust response parameters
temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens=st.sidebar.slider("Max Tokens",min_value=50, max_value=300, value=150)


# main interface for user input
st.write("Ask any question")
user_input=st.text_input("You:")
if user_input:
    response=generate_response(user_input,api_key,llm,temperature,max_tokens)
    st.write(response)
elif user_input:
    st.warning("Please enter the Gemini key in the side bar")
else:
    st.write("Please provide the query")