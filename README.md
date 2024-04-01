# AI Assistant for Data Science 🤖
Welcome to the AI Assistant for Data Science, an interactive tool designed to facilitate and enhance your data science projects. Leveraging the power of AI through OpenAI's APIs and Streamlit's interactive interface, this application allows you to upload CSV files for comprehensive analysis, offering insights, data cleaning suggestions, and exploratory data analysis (EDA) capabilities.
Features
1.	Data Science Journey Initiation: Start your adventure by uploading a CSV file and let the AI dive into the data, providing valuable insights and understandings.
2.	Automated Exploratory Data Analysis (EDA): Get automated steps and guidelines for EDA, helping you understand the dataset comprehensively.
3.	AI-Driven Insights: Utilize AI capabilities for data cleaning, analysis, and generating new feature suggestions, enhancing your data science project's efficiency and depth.
4.	User-Friendly Interface: An intuitive Streamlit interface that makes data science accessible to practitioners of all levels.

# How to Set Up
1.	Clone the Repository:
git clone https://github.com/BobGanti/DAAI
cd DAAI 

2.	Environment Setup: 
Ensure Python 3.8+ and pip are installed. Create a virtual environment and activate it:
python -m venv davenv 
source davenv/bin/activate  # On Windows use `davenv\Scripts\activate` 

3.	Dependencies Installation: 
Install the required libraries using:
pip install -r requirements.txt 

4.	Environment Variables: 
Create a .env.local file at the root of the project directory and add your OpenAI API key:
OPENAI_API_KEY='your_openai_api_key_here' 

5.	Running the Application: 
Launch the Streamlit application:
streamlit run daai.py

# Usage
•	Uploading Data: Navigate to the sidebar and upload your CSV file to start the analysis process.
•	Interface Guide: Use the sidebar to perform various actions and view the main panel for results and insights.
•	Deep Dive Analysis: The application automatically performs data cleaning, summarizes data, identifies outliers, and much more. Use the provided AI tools to explore further and gain deeper insights into your dataset.


# Development
This project is developed with Python, using Streamlit for the frontend and Pandas for data manipulation. The AI-driven analysis is powered by langchain_community.llms and langchain_experimental.agents, offering a robust set of tools for automated data analysis.

# Contributing
Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated. Check our Contributing Guidelines for more information on how to get started.

# License
Distributed under the MIT License. See LICENSE for more information.

# Acknowledgements
•	Streamlit
•	Langchain
•	OpenAI

# .env.local 
This file should contain your OpenAI API key
OPENAI_API_KEY='your_openai_api_key_here'

# requirements.txt
streamlit==latest
pandas==latest
python-dotenv==latest
openai==latest
langchain_community==latest
langchain_experimental==latest

# CONTRIBUTING.md
In this file, you'll want to include:
•	Introduction to contributing to the project
•	Steps for setting up a development environment
•	Guidelines for coding standards
•	Process for submitting pull requests
•	How to report bugs or suggest enhancement
