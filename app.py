import streamlit as st
import pandas as pd
from rag_processor import load_data, setup_agent, GEMINI_MODEL
from data_visualizer import (
    plot_sales_trend, 
    plot_regional_sales, 
    plot_product_performance, 
    plot_customer_segmentation
)
from langchain.memory.buffer import ConversationBufferMemory


st.set_page_config(
    page_title="InsightForge: AI-Powered BI Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)


if 'df' not in st.session_state:
    st.session_state['df'] = load_data('sales_data.csv')
    st.session_state['agent'] = setup_agent(st.session_state['df'])
    st.session_state['memory'] = ConversationBufferMemory(memory_key="chat_history")
    st.session_state['messages'] = [{"role": "assistant", "content": "Hello! I'm InsightForge, your AI BI assistant. Ask me anything about the sales data, like 'What are the top selling regions?' or 'What is the median customer age?'"}]


with st.sidebar:
    st.title("📊 Data Visualizations")
    st.header(f"Data Loaded: {len(st.session_state['df']):,} Rows")
    st.markdown("---")
    st.plotly_chart(plot_sales_trend(st.session_state['df']), use_container_width=True)
    st.plotly_chart(plot_regional_sales(st.session_state['df']), use_container_width=True)
    st.plotly_chart(plot_product_performance(st.session_state['df']), use_container_width=True)
    st.plotly_chart(plot_customer_segmentation(st.session_state['df']), use_container_width=True)

st.title("🤖 InsightForge: AI-Powered BI Assistant")



for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Ask a question about sales data..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # FIXED: .run() for older LangChain versions; use .invoke() if 0.3+
    if st.session_state.agent is not None:
        try:
            with st.spinner('Analyzing data and generating insight...'):
                ai_response = st.session_state.agent.run(prompt)  # Use .run() for 0.2.x compatibility

        except Exception as e:
            ai_response = f"An error occurred during analysis. Please try rephrasing your question. Error: {e}"
    else:
        ai_response = "Data agent is not available. Please ensure 'sales_data.csv' exists and is correctly loaded."

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(ai_response)
        
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": ai_response})