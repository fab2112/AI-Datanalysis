# Prompt
import os
import sys
import traceback
import streamlit as st

from io import StringIO
from colorama import Fore


def define_graph_type(llm: object, question_user: str, hist_questions: str) -> str:
    """
    This function analyzes the question via LLM and sets the graph type for RAG routing.

    Args:
        llm: llm langchain object.
        user_question: last request sent from the user.
        hist_questions: history of messages sent by the user.

    Returns:
        response: text string.
    """

    prompt_template = f"""
    <question>
    Answer the following question with a single specific word.
    {question_user}
    Answer with just one word.
    </question>
    
    <hist_questions>
    {hist_questions}
    </hist_questions>

    Analyze the question within the <question> tags and determine which type of graph is most appropriate for 
    the question, according to the options listed in <allowed_words>.
    
    Analyze message history within the <hist_questions> tags for more context in response.

    <allowed_words>
    answer "treemap_plot" if has in question (treemap)
    answer "sunburst_plot" if has in question (sunburst)
    answer "density_contour_plot" if has in question (contorno de densidade, density contour)
    answer "violin_plot" if has in question (violin, violino)
    answer "candle_plot" if has in question (candle, vela, ohlc, candlestick)
    answer "heatmap" if has in question (heatmap, matriz, matrix)
    answer "scatter_2d_plot" if has in question (scatter, point)
    answer "bubble_plot" if has in question (bolhas, bubble)
    answer "scatter_3d_plot" if has in question (scatter 3d, 3d points)
    answer "surface_plot" if has in question (superficie, volume, surface)
    answer "surface_plot" if has in question (surface, plano, superficie, volume)
    answer "bar_plot"  if has in question (barra, bar, barras, bar)
    answer "line_plot" if has in question (linha)
    answer "histogram_plot" if has in question (histograma, distribuição)
    answer "pie_plot" if has in question (pizza, pie)
    answer "box_plot" if has in question (caixa)
    answer "area_plot" if has in question (area)
    answer "choroplethmap_plot" if has in question (mapa coropletico)
    answer "densitymap_plot" if has in question (mapa de densidade)
    answer "scattermap_plot" if has in question (mapa scatter, mapa de pontos)
    answer "polar_plot" if has in question (radar, polar)
    answer "table" if has in question (tabela com dataframe)
    answer "table_plotly" if has in question (tabela com plotly, plot de tabela)
    answer "base_ref" If there is nothing related to the question.
    </allowed_words>

    Use context information for accuracy in the answer.
    DECLARE ONLY one WORD in response according with <allowed_words> tags.
    
    Response examples:
    question: 'bring a bar plot of the states' 
    response: 'bar_plot'
    
    question: 'traga um plot de barras dos estados' 
    response: 'bar_plot'
    
    """

    response = llm.invoke(prompt_template)

    return response.content.strip()


def process_prompt(
    session_msgs: list[dict], user_question: str, data: dict, llm: object
) -> str:
    """
    This function does the processing, RAG routing and generation of the final prompt.

    Args:
        session_msgs: all requests sent from a session.
        user_question: last request sent from the user.
        data: dictionary of dataframes.
        
    Returns:
        prompt: text string
    """

    # Process data for prompt <metadata>
    try:
        output_parts = []
        # Iterate over the dataframes in the dictionary
        for index, (name, df) in enumerate(data.items(), start=1):
            buffer = StringIO()
            df.info(buf=buffer)
            df_info = buffer.getvalue()
            df_sample = df.sample(st.session_state["sample_var"]).to_csv(
                path_or_buf=None, index=False
            )
            df_name = f"DF_{index}"

            # Add dataframe information to the list of parts
            output_parts.append(
                f"\n<{df_name}>\n[INFO {df_name}]:\n{df_info}[SAMPLES {df_name}]:\n{df_sample}</{df_name}>\n"
            )
        # Join parts
        metadata = "\n".join(output_parts)
    except Exception as e:
        data = {}
        exception_name = type(e).__name__
        track_line = f" L-{traceback.extract_tb(e.__traceback__)[0].lineno}"
        message_ = "WARNING! Sample data error, provided only columns as samples"
        st.error(f"{message_}  <{exception_name}: {track_line}>")
        metadata = str(df.columns)
        # raise sys.exc_info()[0]

    # Code in context
    if st.session_state["context_code_var"]:
        st.session_state["context_code_var"] = False
        context_code = st.session_state["last_code"]
    else:
        context_code = "No code returned for context."

    user_questions = [item for item in session_msgs if item["role"] == "user"]
    questions_text = "\n".join([item["question"] for item in user_questions])

    # Process RAG routing
    graph_type = define_graph_type(llm, user_question, questions_text)
    dir_path = "./ragdata"
    files = os.listdir("./ragdata")
    for file_ in files:
        file_name, file_ext = os.path.splitext(file_)
        if file_name == graph_type and file_ext == ".txt":
            file_path = os.path.join(dir_path, file_)
            with open(
                file_path,
                "r",
            ) as code:
                params_plot = f"\n\n<code_ref>\n{code.read()}\n</code_ref>"
            break

        else:
            with open("ragdata/base_ref.txt") as code:
                params_plot = f"\n\n<code_ref>\n{code.read()}\n</code_ref>"

    # Check if graphic is to be created
    if graph_type in [
        "scatter_2d_plot",
        "bubble_plot",
        "scatter_3d_plot",
        "bar_plot",
        "line_plot",
        "histogram_plot",
        "pie_plot",
        "box_plot",
        "area_plot",
        "choroplethmap_plot",
        "densitymap_plot",
        "scattermap_plot",
        "table_plotly",
        "polar_plot",
        "surface_plot",
        "heatmap",
        "candle_plot",
        "violin_plot",
        "surface_plot",
        "density_contour_plot",
        "sunburst_plot",
        "treemap_plot",
    ]:
        prompt_context = f"""
        For plots, ONLY use the "Plotly" library and bring fig object into the result variable.
        The template should ONLY be “plotly”, when not requested.
        All texts in titles, axis titles, legends, hovers, etc., SET to language "{st.session_state["lang_var"]}".
        SET Legend title according to legend data, when not requested.
        On the x and y axes, place large words or numbers representing dates at 45 degrees, when not requested.
        The title color must follow the template standard, when not requested.
        DEFINE text in "xaxis_title" or "xaxis_title" according to <main_question>, removing symbols and special characters.
        DECLARE the chart title according to <main_question>.
        Follow the underlying tags below EXACTLY, ALWAYS guided by the <main_question> tag.
        SET the update_traces() and update_layout() configs BASED in context within the <code_ref> tags for <main_question>.
        {params_plot}"""

    # Reply in dataframe or string
    elif graph_type == "table":
        prompt_context = """
        Any table or list request by the <main_question> tag, the response must ONLY be a return with
        dataframe and NEVER a graph.
        """

    # Base ref_code
    else:
        with open("ragdata/base_ref.txt") as code:
            params_plot = f"\n\n<code_ref>\n{code.read()}\n</code_ref>"
        prompt_context = f"""
        For plots, ONLY use the "Plotly" library and bring fig object into the result variable.
        {params_plot}"""

    print(f"\n\n{Fore.LIGHTGREEN_EX}STARTING RUNTIME...{Fore.RESET}")

    print(f"\n{Fore.LIGHTBLUE_EX}GRAPH TYPE BASE:{Fore.RESET} {graph_type}")

    # Set prompt language
    if st.session_state["lang_var"] == "en":
        lang = """Respond ONLY in <en> language. Configure all graphics parameters for the <en> language."""
    else:
        lang = """Respond ONLY in <pt-BR> language. Configure all graphics parameters for the <pt-BR> language."""

    prompt_main = f"""
    
        <metadata>
        {metadata}
        </metadata>
        
        
        
        DEFINE DF_* according to the <main_question> using the variables already declared <DF_1, DF_2, DF_*, ...>. 
        A priori assumes DF as DF_1.
        
        ```python
        # TODO: import the necessary dependencies.
        import pandas as pd 
        ...
            
        df = pd.DataFrame(DF_*)
            
        # Complete your code here.
        ...
        
        # bring the result here.
        result = ...
        ```
    
    
        Answer the <main_question> tag concisely and accurately. For greater accuracy in your answers, follow the context EXACTLY
        provided by the <guidelines> and <last_code> tags, always driven by the <main_question> tag request and the data
        in the <metadata> tag. Answers with numbers ONLY with two decimal places. In case of error, BEBUG the code according to
        <message_error> tag and the wrong code tag <code_error>.
                    
                        
        <guidelines>
        Follow the guidelines below and use the <messages_history> and <last_code> tags to improve understanding only as context:        
        {prompt_context}
        </guidelines>
            
            
        <messages_history>
        {questions_text}
        </messages_history>


        <main_question>
        {user_question}
        </main_question>
        
        
        <last_code>
        ```python
        {context_code}
        ```
        </last_code>
        
        

        Variable `GEOJSON: list[dict]` is already declared.

        Generate python code and return full updated code.
        
        {lang}
        
        """

    return prompt_main
