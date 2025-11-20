import pandas as pd
import os
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.agents import AgentType
import streamlit as st



GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

def load_data(file_path: str) -> pd.DataFrame:
    """Loads and preprocesses the sales data."""
    try:
        df = pd.read_csv(file_path)
        
        df['Date'] = pd.to_datetime(df['Date'])
        df['Sales_Per_Age'] = df['Sales'] / df['Customer_Age']
        return df
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}")
        return pd.DataFrame()

def setup_agent(df: pd.DataFrame):
    """
    Sets up the LangChain Pandas DataFrame Agent (the RAG layer).
    """
    if df.empty:
        return None

    # Explicitly pass the API key to the ChatGoogleGenerativeAI constructor.
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.0,
        api_key=GEMINI_API_KEY 
    )
    
    # Tool description
    toolkit_description = (
        "This agent can access and analyze a Pandas DataFrame containing sales data. "
        "It can perform calculations, descriptive statistics (median, std dev), filtering, "
        "and aggregation (e.g., sales by product, age distribution, regional totals) "
        "to extract precise, factual insights directly from the data. "
        "The columns available are: Date, Product, Region, Sales, Customer_Age, "
        "Customer_Gender, Customer_Satisfaction, and Sales_Per_Age."
    )

    
    agent = create_pandas_dataframe_agent(
        llm=llm, 
        df=df, 
        verbose=True,
        
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
        max_iterations=10,
        handle_parsing_errors=True,
        
        allow_dangerous_code=True, 
        agent_executor_kwargs={"handle_parsing_errors": True}
    )
    
    return agent

# Define the model name for display in Streamlit
GEMINI_MODEL = "gemini-2.5-flash"