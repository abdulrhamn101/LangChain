# Imports
import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
import openai
import streamlit.components.v1 as components
openai.organization = "org-VHPYo0VBlaFsLZevBb8m2fNj"

def show_sidebar():
    st.sidebar.title("About Me")
    st.sidebar.markdown("Hello! I'm Nawaf, Senior AI student")
    st.sidebar.markdown("Connect with me on LinkedIn:")
    st.sidebar.markdown('<a href="https://www.linkedin.com/in/nawafbinsaad/" target="_blank">'
                        '<img src="https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Logo.svg.original.svg" width="120">'
                        '</a>', unsafe_allow_html=True)
    st.sidebar.title("Project Description")
    st.sidebar.markdown("Welcome to AskAI App, where data exploration meets AI magic!.")
    st.sidebar.markdown("Upload a CSV file and Prepare to be amazed as you witness the harmony between your questions and the intelligent agent's responses, transforming raw data into a mesmerizing narrative.")

def main():
    counter = 0
    # Load environment variables from .env file
    load_dotenv()

    # Access the API key from the environment variable
    api_key = os.getenv("OPENAI_API_KEY")

    # Title and description
    st.title("AskAI App")
    st.write("Upload a CSV file, then speak with your data!")

    # Show sidebar
    show_sidebar()

    # Upload File
    file =  st.file_uploader("Upload CSV file",type=["csv"])
    if not file: st.stop()

    # Read Data as Pandas
    data = pd.read_csv(file)

    # Display Data Head
    st.write("Original Data Preview:")
    st.dataframe(data)

    # Data Filtering
    st.subheader("Data Filtering")
    st.write("Note: You can skip the filtering step if you prefer to explore the original data.")

    # Create a copy of the original data
    filtered_data = data.copy()

    # Display filter options
    filter_option = st.selectbox("Filter Option \"choose None if you want to skip\"", ["None", "Column Values"])

    # Task: Select Columns for Filtering
    filter_columns = st.multiselect("Select Columns \"select the filter values for each selected column\"", data.columns)

    # Task: Filter Data based on Column Values
    filter_values = {}
    for column in filter_columns:
        values = data[column].unique()
        filter_values[column] = st.selectbox(f"Select Filter Value for '{column}'", values, key=column)

    # Apply filters if any
    if filter_option == "Column Values" and filter_columns:
        filtered_data = filtered_data.copy()
        filtered_mask = pd.Series(True, index=filtered_data.index)
        for column, value in filter_values.items():
            if value:
                column_mask = filtered_data[column].astype(str).str.contains(str(value), na=False)
                filtered_mask &= column_mask
        filtered_data = filtered_data[filtered_mask]

    # Display note about empty data
    if filter_option == "Column Values" and filter_columns and filtered_data.empty:
        st.write("No data matching the filter values.")
    # Display filtered data
    st.subheader("Filtered Data Preview:")
    st.dataframe(filtered_data)

    # Define pandas df agent - 0 ~ no creativity vs 1 ~ very creative
    agent = create_pandas_dataframe_agent(OpenAI(temperature=0.5),data,verbose=True) 
    
    # Define Generated and Past Chat Arrays
    if 'generated' not in st.session_state: 
        st.session_state['generated'] = []

    if 'past' not in st.session_state: 
        st.session_state['past'] = []

    # CSS for chat bubbles
    chat_bubble_style = \
    """
        .user-bubble {
            background-color: dodgerblue;
            color: white;
            padding: 8px 12px;
            border-radius: 15px;
            display: inline-block;
            max-width: 70%;
        }
        
        .gpt-bubble {
            background-color: #F3F3F3;
            color: #404040;
            padding: 8px 12px;
            border-radius: 15px;
            display: inline-block;
            max-width: 70%;
            text-align: right;
        }
    """

    # Apply CSS style
    st.write(f'<style>{chat_bubble_style}</style>', unsafe_allow_html=True)

    # Accept input from user
    query = st.text_input("Ask your data:") 

    # Execute Button Logic
    if st.button("Ask") and query:
        if counter < 3:
            with st.spinner('Generating response...'):
                try:
                    answer = agent.run(query)

                    # Store conversation
                    st.session_state.past.append(query)
                    st.session_state.generated.append(answer)

                    # Display conversation in reverse order
                    for i in range(len(st.session_state.past)-1, -1, -1):
                        st.write(f'<div class="gpt-bubble">{st.session_state.generated[i]}</div>', unsafe_allow_html=True)
                        st.write(f'<div class="user-bubble">{st.session_state.past[i]}</div>', unsafe_allow_html=True)
                        st.write("")

                    counter += 1

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("You have reached the maximum number of queries.")
    # Clear conversation button
    if st.button("Clear Conversation"):
        st.session_state.past = []
        st.session_state.generated = []
    # Export conversation log button
    if st.button("Export Conversation"):
        if len(st.session_state.past) > 0:
            conversation_log = ""
            for i in range(len(st.session_state.past)):
                conversation_log += f"User: {st.session_state.past[i]}\n"
                conversation_log += f"AI Agent: {st.session_state.generated[i]}\n\n"
            st.download_button(
                "Download Conversation",
                data=conversation_log,
                file_name="conversation.txt",
                mime="text/plain"
            )
        else:
            st.error("No conversation to export.")
  
if __name__ == "__main__":
    main()
