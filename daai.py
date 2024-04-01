import os
import openai
import streamlit as st
import pandas as pd
from dotenv import load_dotenv, find_dotenv

from langchain_community.llms import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent


load_dotenv(".env.local")
openai.api_key = os.getenv("OPENAI_API_KEY")

st.title("AI Assistant for Data Science 🤖")
st.write("Hello 👋, how can I help with your data science projects.")

with st.sidebar:
    st.write("""**Your Data Science Adventure Beginis with a csv file.**""")
    st.caption("""*You may already know that every data science journey starts with a dataset. That's why I would love for you to upload a CSV file. Onnce I have your data, I'll dive into understanding it and have some fun while doing so...*""")
    
    st.divider()
    
    st.caption("<p style='text-align:center'>Developed by Bobga</p>", unsafe_allow_html=True)

# Initialise the key in session state
if "clicked" not in st.session_state:
    st.session_state.clicked = {1:False}

# Function to update the value in session state
def clicked(button):
    st.session_state.clicked[button] = True

st.button("Get Started", on_click=clicked, args=[1])
if st.session_state.clicked[1]:
    st.header("Data Analysis")
    st.subheader("Solution")

    user_csv = st.file_uploader("Upload your csv file here", type="csv")
    if user_csv is not None:
        user_csv.seek(0)  # Ensuring that the pointer is at the start of file
        df = pd.read_csv(user_csv, low_memory=False)

        llm = OpenAI(temperature=0)

        # Sidebar function 
        @st.cache_data
        def eda_steps():
             eda_steps = llm("What are the steps of Exploratory Data Analysis (EDA)")
             return eda_steps
        
        # Instantiate a AI agent
        pandas_agent = create_pandas_dataframe_agent(llm, df, verbose=True)

        # Function of the main scripts949
        @st.cache_data
        def function_agent():
            st.write("**Data Overview**")
            st.write("The firs rows of the dataset, and the shape, looks like this:")
            st.write(df.head(), df.shape)

            st.write("**Data Cleaning**")
            columns_df = pandas_agent.run("What are the meanings of the columns?")
            st.write(columns_df)
            missing_values = pandas_agent.run("How many missing values are in this dataframe have? Start the answer with 'There are'")
            st.write(missing_values)
            duplicates = pandas_agent.run("Are there any duplicate values and if so where?")
            st.write(duplicates)

            st.write("**Data Summarisation**")
            st.write(df.describe())
            correlation_analysis = pandas_agent.run("Calculate the correlations between the numerical variables to identify potential relationships.")
            st.write(correlation_analysis)
            outliers = pandas_agent.run("Identify Outliers in the data that may be errorneous or that may have a significant impact on the analysis.")
            st.write(f"**Outliers:** {outliers}")
            new_features = pandas_agent.run("What new features would be interesting to create for this analysis?")
            st.write(f"**Feature Selection:** {new_features}")
            return
        
        @st.cache_data
        def function_question_variable():
             st.line_chart(df, y = [user_question_variable])
             summary_statistics = pandas_agent.run(f"What are the mean, median, mode, standard deviation, variance, range, quartiles, skewness and kurtosis of {user_question_variable}")
             st.write(summary_statistics)
             normality = pandas_agent.run(f"Check for normality or specific distribution shapes of {user_question_variable}")
             st.write(normality)
             outliers = pandas_agent.run(f"Assess the presence of outliers of {user_question_variable}")
             st.write(f"**Outliers:** {outliers}")
             trends = pandas_agent.run(f"Analyse trends, seasonality, and cyclic patters of {user_question_variable}")
             st.write(f"**Trends:** {trends}")
             missing_values = pandas_agent.run(f"Determin the extend of missing values of {user_question_variable}")
             st.write(f"**Missing Values:** {missing_values}")
             return
        
        @st.cache_data
        def function_question_dataframe():
            dataframe_info = pandas_agent.run(user_question_dataframe)
            st.write(dataframe_info)
            return
        
        st.header("Exploratory Data Analysis")
        st.subheader("General information about the dataset")

        with st.sidebar:
            st.caption("<p style='text-align:center;font-size:20px;'>Menu</p>", unsafe_allow_html=True)

            with st.expander("Steps of EDA"):
                st.write(llm(eda_steps()))
            
        function_agent()

        st.subheader("Variable to study")
        user_question_variable = st.text_input("What variable are you interested in")
        if user_question_variable is not None and user_question_variable != "":
            function_question_variable()

            st.subheader("Further study")
        if user_question_variable:
            user_question_dataframe = st.text_input("Is there anything else?")
            if user_question_dataframe is not None and user_question_dataframe !=""and        user_question_dataframe not in ("","no","No","NO"):
                function_question_dataframe()
            else: st.write("")

