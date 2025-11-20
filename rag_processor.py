import pandas as pd
import os
import streamlit as st
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI


try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:

    st.error("FATAL ERROR: Gemini API Key not found in Streamlit secrets or environment.")

def load_data(file_path: str) -> pd.DataFrame:
    """Loads and preprocesses the sales data."""
    if not GEMINI_API_KEY: return pd.DataFrame()
    try:
        df = pd.read_csv(file_path)
        
        df['Date'] = pd.to_datetime(df['Date'])
        df['Sales_Per_Age'] = df['Sales'] / df['Customer_Age']
        return df
    except FileNotFoundError:
        st.error(f"Error: Data file not found at {file_path}")
        return pd.DataFrame()

def setup_agent(df: pd.DataFrame):
    """
    Sets up the LangChain Pandas DataFrame Agent (the RAG layer).
    """
    if df.empty or not GEMINI_API_KEY:
        return None

   
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.0,
        api_key=GEMINI_API_KEY 
    )
    
   
    
    agent = create_pandas_dataframe_agent(
        llm=llm, 
        df=df, 
        verbose=True,
        

        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
        max_iterations=10,
        
       
        allow_dangerous_code=True, 


        handle_parsing_errors=True,
    )
    
    return agent


GEMINI_MODEL = "gemini-2.5-flash"