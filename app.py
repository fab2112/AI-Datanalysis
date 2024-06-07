# App
import traceback
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objs as go
import plotly.express as px

from colorama import Fore
from agent import AgentAI
from prompt import process_prompt
from styles import process_styles

from langchain_groq import ChatGroq
from langchain_cohere import ChatCohere
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama



# Parameters
WHITELIST_ENV = ["json", "statsmodels", "scipy", "datetime"]
N_SAMPLES = 5  # Number of samples sent to the prompt in non-private mode
MAX_ATTEMPTS = 10  # Number of calls to LLM in case of execution error
VERBOSE = False  # Show final prompt and errors



# Session_state reruns variables
if "last_code" not in st.session_state:
    st.session_state["last_code"] = ""
if "context_code_var" not in st.session_state:
    st.session_state["context_code_var"] = False
if "count_var" not in st.session_state:
    st.session_state["count_var"] = 1
if "lang_var" not in st.session_state:
    st.session_state["lang_var"] = "en"
if "response_error_var" not in st.session_state:
    st.session_state["response_error_var"] = ""
if "sample_var" not in st.session_state:
    st.session_state["sample_var"] = 0   
if "agent_var" not in st.session_state:
    st.session_state["agent_var"] = None 


def clear_chat_history() -> None:
    """
    Clear chat history | reset session variables | stop runtime agent.
    """

    # Reset session_state vars
    st.session_state["last_code"] = ""
    st.session_state["context_code_var"] = False
    st.session_state["response_error_var"] = ""
    st.session_state["messages"] = []

    # Stop agent runtime
    agent = st.session_state["agent_var"]
    if agent is not None:
        agent.chat_stop()
        del agent
    
    
def process_chat(llm_agent: AgentAI, llm: object, data: dict) -> None:
    """
    This function creates and processes the chat engine.

    Args:
        llm_agent: AgenteAI object.
        llm: llm langchain object.
        data: dictionary of dataframes
    """

    with st.chat_message("assistant"):
        st.write("Hello, I'm ready to help explore your data. Let's go? 😏")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Show messages from chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if "question" in message:
                st.markdown(message["question"])
            elif "response" in message:
                if isinstance(message["response"], str):
                    st.write(message["response"])
                elif isinstance(message["response"], go.Figure):
                    config = {"displaylogo": False}
                    st.plotly_chart(message["response"], config=config)
                elif isinstance(message["response"], list) or isinstance(message["response"], tuple):
                    for i in range(len(message["response"])):
                        if isinstance(message["response"][i], go.Figure):
                            config = {"displaylogo": False}
                            st.plotly_chart(message["response"][i], config=config)
                        else:
                            st.write(message["response"][i])          
                elif isinstance(message["response"], dict):
                    for key_, value_ in message["response"].items():
                        if isinstance(message["response"][key_], go.Figure):
                            config = {"displaylogo": False}
                            st.plotly_chart(message["response"][key_], config=config)
                        else:
                            st.write(message["response"][key_])                       
                else:
                    st.write(message["response"])
            elif "error" in message:
                st.text(message["error"])

    # Display the expander with code
    _, col1, _ = st.columns([1, 8, 1])
    if st.session_state["last_code"]:
        with col1:
            with st.expander("Python code"):
                st.code(st.session_state["last_code"], language="python")
    elif st.session_state["response_error_var"] != "":
        with col1:
            with st.expander("Response error"):
                st.error(st.session_state["response_error_var"])

    # Check if message in chat_input
    if user_question := st.chat_input(
        "Bring context for better accuracy in responses."
    ):
        # Reset session_state vars
        st.session_state["response_error_var"] = ""

        user_question = user_question.strip()
        st.session_state.messages.append({"role": "user", "question": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                
                try:
                    prompt = process_prompt(st.session_state.messages, user_question, data, llm)
                    response = llm_agent.chat(prompt)
                except Exception as e:
                    exception_name = type(e).__name__
                    track_line = (f" L-{traceback.extract_tb(e.__traceback__)[0].lineno}")
                    response = f"EXCEPTION ERROR: {exception_name}: {track_line}"
                    # raise sys.exc_info()[0]

                st.session_state["last_code"] = llm_agent.get_last_code()
                # Code in context
                st.session_state["context_code_var"] = True

                # Convert output with numbers (int and float) to string
                response = (
                    str(response)
                    if isinstance(response, int) or isinstance(response, float)
                    else response
                )
                # Convert nd.array output to string
                response = (
                    str(response.item())
                    if isinstance(response, np.ndarray)
                    else response
                )

                print(f"\n{Fore.LIGHTYELLOW_EX}CODE RESPONSE:{Fore.RESET}", type(response))

                try:
                    if isinstance(response, str):
                        # Handles exceptions
                        if response.split()[0] == "EXCEPTION":
                            st.session_state["response_error_var"] = response
                            response = "I'm sorry, I was unable to fulfill your request 😕, \
                                I suggest clearing your message history and trying again."
                            # Hide last code
                            st.session_state["last_code"] = None
                        st.session_state.messages.append(
                            {"role": "assistant", "response": response}
                        )
                        
                    else:
                        st.session_state.messages.append(
                            {"role": "assistant", "response": response}
                        )
                except Exception as e:
                    exception_name = type(e).__name__
                    track_line = (
                        f" L-{traceback.extract_tb(e.__traceback__)[0].lineno}"
                    )
                    message_ = "Response fail:"
                    st.error(f"{message_}  <{exception_name}: {track_line}>")
                    # raise sys.exc_info()[0]

                st.rerun()


def get_llm_agent(data: dict, llm: object) -> AgentAI:
    """
    The function creates an agent with the dataframes extracted from the files.

    Args:
        data: a dictionary with dataframes extracted from the sent data.
        llm: llm langchain object
        
    Returns: object AgentAI
    """
    
    agent = AgentAI(
        data=list(data.values()),
        llm=llm,
        max_attempts=MAX_ATTEMPTS,
        whitelist=WHITELIST_ENV,
        verbose=VERBOSE,
    )
    
    st.session_state["agent_var"] = agent

    return agent


def extract_dataframes(raw_files: list) -> dict:
    """
    This function extracts data from the loaded files and converts it into a dictionary.

    Args:
        raw_files: upload_File object.
        
    Returns:
        dfs: dictionary with dataframes.
    """

    dfs = {}

    for raw_file in raw_files:
        if raw_file.name.split(".")[1] == "csv":
            var = st.session_state["count_var"]
            csv_name = f"DF_{var}___{raw_file.name.split('.')[0]}"
            df = pd.read_csv(raw_file, encoding="utf-8", encoding_errors="replace")

            dfs[csv_name] = df
            st.session_state["count_var"] += 1

        elif (raw_file.name.split(".")[1] == "xlsx") or (
            raw_file.name.split(".")[1] == "xls"
        ):
            var = st.session_state["count_var"]
            # Read the Excel file
            xls = pd.ExcelFile(raw_file)

            # Iterate through each sheet in the Excel file and store them into dataframes
            for index, sheet_name in enumerate(xls.sheet_names):
                var = st.session_state["count_var"]
                dfs[f"DF_{var}___Sheet-{index}__{sheet_name}"] = pd.read_excel(
                    raw_file, sheet_name=sheet_name
                )
                st.session_state["count_var"] += 1

    return dfs


def main() -> None:
    """
    Main function as entry point for the script.
    """

    st.set_page_config(
        layout="wide",
        menu_items={"About": "https://github.com/fab2112/AI-Datanalysis"},
        initial_sidebar_state="expanded",
        page_icon="📊",
        page_title="AI-Datanalysis",
    )

    process_styles()

    # Header title
    header = st.container(border=True)
    header.header("AI-Datanalysis", divider="violet")

    with st.sidebar:
        st.header("Settings:", divider="grey")

        # Button to load data
        side_container_2 = st.sidebar.container(border=True)

        # Button to delete chat history
        side_container_2.button("🗑️ &nbsp;&nbsp;Clear chat", on_click=clear_chat_history)

        try:
            file_upload = side_container_2.file_uploader(
                " 📂 &nbsp;Load data",
                accept_multiple_files=True,
                type=["csv", "xls", "xlsx"],
            )
        except Exception as e:
            exception_name = type(e).__name__
            track_line = f" L-{traceback.extract_tb(e.__traceback__)[0].lineno}"
            message_ = "File_upload failed:"
            st.error(f"{message_}  <{exception_name}: {track_line}>")
            # raise sys.exc_info()[0]

        # Reset dataframe count state variable
        st.session_state["count_var"] = 1

    if len(file_upload) > 0:
        try:
            data = extract_dataframes(file_upload)

            _, col3, _ = st.columns([0.1, 3, 0.1])

            with col3:
                df = st.selectbox(
                    "Data loaded successfully!", tuple(data.keys()), index=0
                )
                st.dataframe(data[df], use_container_width=True)
        except Exception as e:
            data = {}
            exception_name = type(e).__name__
            track_line = f" L-{traceback.extract_tb(e.__traceback__)[0].lineno}"
            message_ = "Data display failed:"
            st.error(f"{message_}  <{exception_name}: {track_line}>")
            # raise sys.exc_info()[0]

        with st.sidebar:
            # Select language for prompt
            languages = {"English 🇬🇧": "en", "Portuguese 🇧🇷": "pt-BR"}
            samples = {"Anonymous": 0, "Some Samples": N_SAMPLES,}
            with st.expander("Prompt language & privacy"):
                sel_lang = st.radio(
                    "Main language of the chat",
                    options=languages,
                    horizontal=False,
                    key="selected_language",
                )
                st.session_state["lang_var"] = languages[sel_lang]
                
                privacy_var = st.radio(
                    "Data privacy in the prompt",
                    options=samples,
                    horizontal=True,
                    key="selected_samples",
                )
                st.session_state["sample_var"] = samples[privacy_var]

            # Button to select models
            side_container_3 = st.sidebar.container(border=True)
            llm_type = side_container_3.selectbox(
                "⚙️ &nbsp;Proprietary model",
                (
                    "Groq_llama3-70B",
                    "Groq_mixtral-8x7B-32768",
                    "Google_gemini-1.5-pro",
                    "Google_gemini-1.5-flash",
                    "OpenAI_gpt-3.5-turbo",
                    "Cohere_command-r-plus",
                ),
                index=0,
            )

            api_key = side_container_3.text_input(
                "🔑 &nbsp;API-KEY ", type="password", key="password_1"
            )

            # Temperature slider
            llm_temp = side_container_3.slider("🌡️ &nbsp;Temperature", 0.0, 1.0, 0.0, key="temp_1")
            # Advanced settings
            with side_container_3.expander("Advanced Settings"):
                llm_top_p = st.slider("Top-p", 0.0, 1.0, 1.0)
            
            # Local model   
            with st.expander("⚙️ &nbsp;Local Model"):
                ollama_model = st.text_input("&nbsp;Ollama Model ",  key="ollama_model")
                ollama_url = st.text_input("&nbsp;Ollama URL ",  key="url_2")
                # Temperature slider
                ollama_temp = st.slider("🌡️ &nbsp;Temperature", 0.0, 1.0, 0.0, key="temp_2")
                

            with st.expander("Maps settings"):
                uploaded_json_file = st.file_uploader(
                    "📂 &nbsp;Load Geojson data",
                    accept_multiple_files=False,
                    type=["json"],
                )
                if uploaded_json_file is not None:
                    with open("./maps_configs/geojson_data.json", "wb") as f:
                        f.write(uploaded_json_file.getbuffer())

                mapbox_token = st.text_input(
                    "🔑 &nbsp;Mapbox token ", type="password", key="password_2"
                )
                # Load Mapbox token in global Plotly
                px.set_mapbox_access_token(mapbox_token)

        if api_key != "":
            # Google
            if (
                llm_type == "Google_gemini-1.5-pro"
                or llm_type == "Google_gemini-1.5-flash"
            ):
                model = (
                    "gemini-1.5-flash-latest"
                    if llm_type == "Google_gemini-1.5-flash"
                    else "gemini-1.5-pro-latest"
                )
                llm = ChatGoogleGenerativeAI(
                    model=model,
                    google_api_key=api_key,
                    temperature=llm_temp,
                    top_p=llm_top_p,
                )

            # Groq
            elif llm_type == "Groq_llama3-70B" or llm_type == "Groq_mixtral-8x7B-32768":
                model = (
                    "llama3-70b-8192"
                    if llm_type == "Groq_llama3-70B"
                    else "mixtral-8x7b-32768"
                )
                llm = ChatGroq(
                    model_name=model,
                    api_key=api_key,
                    temperature=llm_temp,
                    # max_tokens=8192,
                    # top_p=llm_top_p,
                )

            # Cohere
            elif llm_type == "Cohere_command-r-plus":
                llm = ChatCohere(
                    model="command-r-plus",
                    cohere_api_key=api_key,
                    temperature=llm_temp,
                    max_tokens=5000,
                    top_p=llm_top_p,
                )

            # OpenAI
            elif llm_type == "OpenAI_gpt-3.5-turbo":
                model = "gpt-3.5-turbo"
                llm = ChatOpenAI(
                    model=model,
                    api_key=api_key,
                    temperature=llm_temp,
                    max_tokens=4000,
                )

            # Instantiate PandasAI agent
            agent = get_llm_agent(data, llm)

            # Start the chat
            process_chat(agent, llm, data)
        
        else:
            if ollama_url:
                llm = ChatOllama(model=ollama_model,
                                 temperature=ollama_temp,
                                 base_url=ollama_url)
                
                # Instantiate PandasAI agent
                agent = get_llm_agent(data, llm)

                # Start the chat
                process_chat(agent, llm, data)
            


if __name__ == "__main__":
    main()
