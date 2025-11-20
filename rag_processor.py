import pandas as pd
import os
import streamlit as st
# FINAL FIX: Import AgentType from the base langchain package (Resolves Pylance warning)
from langchain.agents.agent_types import AgentType 
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI


# Assuming you have set your GEMINI_API_KEY securely in .streamlit/secrets.toml
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    # If secrets fail, try environment variable (for local testing/alternate setup)
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    # This ensures the app doesn't crash if the key is missing on deployment
    st.error("FATAL ERROR: Gemini API Key not found in Streamlit secrets or environment.")
    

def load_data(file_path: str) -> pd.DataFrame:
    """Loads and preprocesses the sales data."""
    if not GEMINI_API_KEY: return pd.DataFrame()
    try:
        df = pd.read_csv(file_path)
        # Convert to datetime and create a helper column for LLM analysis
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

    # Explicitly pass the API key to the ChatGoogleGenerativeAI constructor.
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.0,
        api_key=GEMINI_API_KEY 
    )
    
    # Tool description is automatically included by the agent
    
    agent = create_pandas_dataframe_agent(
        llm=llm, 
        df=df, 
        verbose=True,
        
        # Using the standard REACT agent for complex reasoning on data
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
        max_iterations=10,
        
        # CRITICAL: Allows the LLM to execute Python code on the DataFrame
        allow_dangerous_code=True, 

        # Removed deprecated/unsupported parameters:
        # handle_parsing_errors=True is handled internally
        # agent_executor_kwargs={"handle_parsing_errors": True} is unsupported
    )
    
    return agent

# Define the model name for display in Streamlit
GEMINI_MODEL = "gemini-2.5-flash"