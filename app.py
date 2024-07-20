import time
import os

from google.cloud import bigquery
import streamlit as st
from vertexai.generative_models import FunctionDeclaration, GenerativeModel, Part, Tool
import vertexai
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory
from vertexai.preview import reasoning_engines

from langfuse.callback import CallbackHandler

os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-ffe47943-a70f-4938-8404-f52b4f82cc3c"
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-45237257-36f1-489f-b0c1-ff7c6be98c23"
os.environ["LANGFUSE_HOST"] = "https://langfuse-ghgzraehsq-uc.a.run.app"

langfuse_handler = CallbackHandler()
langfuse_handler.auth_check()

BIGQUERY_DATASET_ID = "thelook_ecommerce"

def list_datasets():
    """
    List datasets in the project.
    """
    response = client.list_datasets(include_all=True)
    print(response)

    return response

def list_tables(
        dataset_id: str,
        ):
    """
    List tables in a dataset.
    """
    response = client.list_tables(dataset_id)
    print(response)

    return response

def get_table(
        table_id: str,
        ):
    """
    Get information about a table.
    """
    response = client.get_table(table_id)
    print(response)

    return response

def sql_query(
        query: str,
        ):
    """
    Get information from data in BigQuery using SQL queries
    """
    response = client.query(query)
    print(response)

    return response

vertexai.init()

safety_settings = {
    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}

model_kwargs = {
    "temperature": 0.28,
    "max_output_tokens": 1000,
    "top_p": 0.95,
    "top_k": 40,
    "safety_settings": safety_settings,
}

model = "gemini-1.0-pro"

st.set_page_config(
    page_title="SQL Talk with BigQuery",
    page_icon="vertex-ai.png",
    layout="wide",
)

col1, col2 = st.columns([8, 1])
with col1:
    st.title("SQL Talk with BigQuery")
with col2:
    st.image("vertex-ai.png")

st.subheader("Powered by Function Calling in Gemini")

st.markdown(
    "[Source Code](https://github.com/GoogleCloudPlatform/generative-ai/tree/main/gemini/function-calling/sql-talk-app/)   •   [Documentation](https://cloud.google.com/vertex-ai/docs/generative-ai/multimodal/function-calling)   •   [Codelab](https://codelabs.developers.google.com/codelabs/gemini-function-calling)   •   [Sample Notebook](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/intro_function_calling.ipynb)"
)

with st.expander("Sample prompts", expanded=True):
    st.write(
        """
        - What kind of information is in this database?
        - What percentage of orders are returned?
        - How is inventory distributed across our regional distribution centers?
        - Do customers typically place more than one order?
        - Which product categories have the highest profit margins?
    """
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if "agent" not in st.session_state:
    st.session_state.agent = reasoning_engines.LangchainAgent(
            model=model,  # Required.
            tools=[
                list_datasets,
                list_tables,
                get_table,
                sql_query,
                ],  # Optional.
            agent_executor_kwargs={
                "verbose": True,
                "return_intermediate_steps": True,
                "max_iterations": 10,
                "max_execution_time": 60,
            },
            model_kwargs=model_kwargs,  # Optional.
        )

# React to user input
if prompt := st.chat_input("Ask me about information in the database..."):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        client = bigquery.Client()

        prompt += """
            Please give a concise, high-level summary followed by detail in
            plain language about where the information in your response is
            coming from in the database. Only use information that you learn
            from BigQuery, do not make up information.
            """
        
        agent_response = st.session_state.agent.query(
            input=prompt,
            config={"callbacks":[langfuse_handler]},
        )

        st.markdown(agent_response['output'])


    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": agent_response['output'],
        }
    )
