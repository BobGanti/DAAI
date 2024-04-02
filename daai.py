import os
import openai
import streamlit as st
import pandas as pd
from dotenv import load_dotenv, find_dotenv

from langchain_community.llms import OpenAI
from langchain.chains.llm import LLMChain
from langchain.chains.sequential import SequentialChain
from langchain.agents.agent_types import AgentType
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent


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
        df = pd.read_csv(user_csv)

        llm = OpenAI(temperature=0)

        # Sidebar function 
        @st.cache_data
        def eda_steps():
             eda_steps = llm("What are the steps of Exploratory Data Analysis (EDA)")
             return eda_steps
        
        # Instantiate a AI agent
        pandas_agent = create_pandas_dataframe_agent(llm, df, verbose=True)

        # Function of the main scripts
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
            st.write(f"**Correlation:** {correlation_analysis}")
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
        
        @st.cache_resource
        def wikipedia(prompt):
            wikipedia_research = WikipediaAPIWrapper().run(prompt)
            return wikipedia_research
  
        @st.cache_data
        def prompt_templates():
            data_problem_template = PromptTemplate(
                input_variables=['business_problem'],
                template='Convert the following business problem into a data science problem: {business_problem}.'
            )

            model_selection_template = PromptTemplate(
                input_variables=['data_problem', 'wikipedia_research'],
                template='Give a list of machine learning algorithms that are suitable to solve this problem: {data_problem}, while using this Wikipedia research: {wikipedia_research}.'
            )

            return data_problem_template, model_selection_template

        @st.cache_resource
        def chains():
            data_problem_chain = LLMChain(llm=llm, prompt=prompt_templates()[0], verbose=True, output_key='data_problem')
            model_selection_chain = LLMChain(llm=llm, prompt=prompt_templates()[1], verbose=True, output_key='model_selection')
            sequential_chain = SequentialChain(chains=[data_problem_chain, model_selection_chain], input_variables=['business_problem', 'wikipedia_research'], output_variables=['data_problem', 'model_selection'], verbose=True)
            return sequential_chain

        @st.cache_data
        def chains_output(prompt, wikipedia_research):
            my_chain = chains()
            my_chain_output = my_chain({
                "business_problem":prompt, 
                "wikipedia_research":wikipedia_research
            })
            my_data_problem = my_chain_output["data_problem"]
            my_model_selection = my_chain_output["model_selection"]

            return my_data_problem, my_model_selection

        @st.cache_data
        def list_to_selectbox(my_model_selection_input):
            algorithm_lines = my_model_selection_input.split('\n')
            algorithms = [algorithm.split(':')[-1].split('.')[-1].strip() for algorithm in algorithm_lines if algorithm.strip()]
            algorithms.insert(0, "Select Algorithm")
            formatted_list_output = [f"{algorithm}" for algorithm in algorithms if algorithm]
            return formatted_list_output

        @st.cache_resource
        def python_agent():
            agent_executor = create_python_agent(
                llm=llm,
                tool=PythonREPLTool(),
                verbose=True,
                max_execution_time=5,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                handle_parsing_errors=True,
            )
            return agent_executor
        
        @st.cache_data
        def python_solution(my_data_problem, selected_algorithm, user_csv):
            solution = python_agent().run(
                f"Write a Python script to solve this: {my_data_problem}, using this algorithm: {selected_algorithm}, using this as your dataset: {user_csv}."
            )
            return solution


        #******** Main *****************#
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
            if user_question_dataframe is not None and user_question_dataframe !=""and user_question_dataframe not in ("","no","No","NO"):
                function_question_dataframe()
            else: st.write("")

            if user_question_dataframe:
                st.divider()
                st.header("Data Science Problem")
                st.write("Now that we have a solid grasp of the data at hand and a clear understanding of the variable we intend to investigate, it's important that to reframe the business problem into a data science problem.")
                
                prompt = st.text_area("Add your prompt here")

                if prompt:
                    wikipedia_research = wikipedia(prompt)
                    my_data_problem = chains_output(prompt, wikipedia_research)[0]
                    my_model_selection = chains_output(prompt, wikipedia_research)[1]

                    st.write(f"\n{my_data_problem}")
                    st.subheader("The ranked list below are algorithms suitable for this task:")
                    st.write(my_model_selection)

                    formatted_list = list_to_selectbox(my_model_selection)
                    selected_algorithm = st.selectbox("Select ML Algorithm", formatted_list)
                    
                    if selected_algorithm is not None and selected_algorithm != "Select Algorithm":
                        st.subheader("Solution")
                        solution = python_solution(my_data_problem, selected_algorithm, user_csv)
                        st.write(solution)
          

