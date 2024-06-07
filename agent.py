# AgentAI
import re
import sys
import json
import builtins
import traceback
import streamlit as st
import pandas as pd
import plotly.graph_objs as go

from time import sleep
from typing import Union
from colorama import Fore
from abc import abstractmethod


WHITELIST_DEFAULT = ["plotly", "numpy", "pandas", "sklearn"]


class AgentAI:
    """
    The class creates an LLM agent for execute python code in a safety environment.

    Attributes:

        data: list[pd.DataFrame]
            A list of pandas DataFrames containing the data to be processed.
        llm: object
            The instantiated large language model (LLM) used to process the data.
        max_attempts: int
            The maximum number of attempts allowed for execution.
        whitelist: list[str]
            A list of modules allowed to run in the environment.
        verbose: bool, optional
            If set to True, enables detailed. Default is False.
        break_run: bool
            Stop atual chat runtime.
        last_prompt: str
            The last prompt send to llm.

    Methods:

        chat(prompt: str)
            Process chat and return code esecution response.
        chat_stop()
            Stop atual chat runtime.
        get_last_code()
            Return the last llm response code.
        get_last_prompt()
            Return the last prompt.
    """

    def __init__(
        self,
        data: list[pd.DataFrame],
        llm: object,
        max_attempts: int,
        whitelist: list[str],
        verbose: bool = False,
    ):
        self.data = data
        self.llm = llm
        self.max_attempts = max_attempts
        self.whitelist = whitelist
        self.break_run = False
        self.verbose = verbose
        self.last_prompt = None
        self.last_code = ""

    def __del__(self):
        pass


    def chat(self, prompt: str) -> Union[list, str, pd.DataFrame, go.Figure]:
        """
        Invoke the chat language model with the provided prompt.
        Execute the returned code from the chat model.
        Return the results of the execution code in one of the defined types.

        Args:
            prompt (str): the name of the module to import.

        Returns:
            Union[list, str, pd.DataFrame, go.Figure]: The output of the chat execution, 
                which could be a list, a string, a Pandas DataFrame or a Plotly figure.
        """

        attempts_var = 0
        prompt_ = prompt

        while attempts_var <= self.max_attempts:
            self.last_prompt = prompt_

            if self.break_run:
                self.break_run = False
                print(f"\n{Fore.LIGHTRED_EX}STOPPED INSTANCE!!!{Fore.RESET}\n")
                break

            if self.verbose:
                print(f"\n{Fore.LIGHTYELLOW_EX}FINAL PROMPT:{Fore.RESET}{prompt_}\n")

            attempts_var += 1
            print(f"\n{Fore.LIGHTBLUE_EX}TRYING...{attempts_var}{Fore.RESET}\n")

            try:
                # Invocar o LLM
                response = self.llm.invoke(prompt_)

                if response is not None and isinstance(response.content, str):
                    match = re.search(r"```python(.*?)```", response.content, re.DOTALL)

                    # Check if has python code returned
                    if match:
                        code = match.group(1)
                    else:
                        code = 'raise Exception("No code returned, try again.")'

                    self.last_code = code

                    code_result = self.exec_code(code)
                    return code_result

                else:
                    print(f"RESPONSE FAIL: TRY-{attempts_var}")

            except Exception as e:
                
                # More randomness for error correction
                if attempts_var > self.max_attempts / 2:
                    self.llm.temperature = 0.5

                error_message = str(e)
                exception_type = f"EXCEPTION_TYPE: {type(e).__name__}\n"
                exception_track = f"EXCEPTION_TRACK: {traceback.format_list(traceback.extract_tb(e.__traceback__)[-1:])[0].strip()}\n"
                exception_message = f"EXCEPTION_MESSAGE: {str(e)}"
                exception_msg = f"{exception_type}{exception_track}{exception_message}"
                
                if self.verbose:
                    print(f"{Fore.LIGHTRED_EX}\nExecution error:\n{error_message}{Fore.RESET}\n")

                # Set line in error code
                lines = self.last_code.split("\n")
                formatted_lines = [
                    f"|Line-{i+1:03}| {line}" for i, line in enumerate(lines)
                ]
                code_withlines = "\n".join(formatted_lines)

                tag_last_code = (
                    f"\n<code_error>\n```python\n{code_withlines}```\n</code_error>\n"
                )
                tag_error = f"\n<message_error>\n{exception_msg}\n</message_error>\n"
                # raise sys.exc_info()[0]
                if len(error_message.split()) > 1:
                    if error_message.split()[1] == "SAFETY:":
                        return error_message
                        
                prompt_ = prompt + tag_last_code + tag_error

                if attempts_var == self.max_attempts:
                    self.llm.temperature = 0.0
                    return f"EXCEPTION ERROR: {error_message}"
                sleep(3)


    def chat_stop(self):
        """
        Stop atual chat runtime by set the attribute break_run to True.
        """
        self.break_run = True


    def get_last_code(self) -> str:
        """
        load and return the last executed code.

        Returns:
            str: string of last code.
        """
        return self.last_code


    def get_last_prompt(self) -> str:
        """
        load and return the last prompt.

        Returns:
            str: string of last prompt.
        """
        return self.last_prompt

    @abstractmethod
    def restricted_import(self, name, globals=None, locals=None, fromlist=(), level=0):
        """
        Imports a module with restrictions based on a whitelist.

        This method overrides the built-in `__import__` function to restrict module imports to a specified whitelist.
        It ensures that only allowed modules and their submodules can be imported. If an attempt is made to import a
        module not in the whitelist, an ImportError is raised.

        Args:
            name (str): The name of the module to import.
            globals (dict, optional): The global variables. Defaults to None.
            locals (dict, optional): The local variables. Defaults to None.
            fromlist (tuple, optional): Names to import from the module. Defaults to ().
            level (int, optional): The level to determine if it's a relative or absolute import. Defaults to 0.

        Raises:
            ImportError: If the module is not in the allowed whitelist.

        Returns:
            module: The imported module if it is allowed.
        """

        allowed_modules = (
            WHITELIST_DEFAULT
            if not self.whitelist
            else WHITELIST_DEFAULT + self.whitelist
        )
        # Allow module and submodules
        if not any(
            name == mod or name.startswith(f"{mod}.") for mod in allowed_modules
        ):
            raise ImportError(
                f"EXCEPTION SAFETY: importing the module '{name.split('.')[0]}' is restricted, is not in whitelist."
            )
        return builtins.__import__(name, globals, locals, fromlist, level)

    @abstractmethod
    def create_isolated_env(self) -> dict:
        """
        Creates an isolated execution environment with restricted built-ins and pre-defined variables.

        This method sets up a global execution environment with a limited set of built-in functions and objects.
        It includes a predefined list of allowed built-ins and custom variables, and restricts the import function
        to use `restricted_import`.

        Returns:
            dict: A dictionary representing the isolated global environment.
        """

        # Defining the necessary builtins
        allowed_builtins = {
            "abs": abs,
            "all": all,
            "any": any,
            "ascii": ascii,
            "bin": bin,
            "bool": bool,
            "bytearray": bytearray,
            "bytes": bytes,
            "callable": callable,
            "chr": chr,
            "classmethod": classmethod,
            "complex": complex,
            "delattr": delattr,
            "dict": dict,
            "dir": dir,
            "divmod": divmod,
            "enumerate": enumerate,
            "filter": filter,
            "float": float,
            "format": format,
            "frozenset": frozenset,
            "getattr": getattr,
            "hasattr": hasattr,
            "hash": hash,
            "help": help,
            "hex": hex,
            "id": id,
            "int": int,
            "isinstance": isinstance,
            "issubclass": issubclass,
            "iter": iter,
            "len": len,
            "list": list,
            "locals": locals,
            "map": map,
            "max": max,
            "memoryview": memoryview,
            "min": min,
            "next": next,
            "object": object,
            "oct": oct,
            "ord": ord,
            "pow": pow,
            "property": property,
            "range": range,
            "repr": repr,
            "reversed": reversed,
            "round": round,
            "set": set,
            "setattr": setattr,
            "slice": slice,
            "sorted": sorted,
            "staticmethod": staticmethod,
            "str": str,
            "sum": sum,
            "super": super,
            "tuple": tuple,
            "type": type,
            "vars": vars,
            "zip": zip,
            "Exception": Exception,
            "__import__": self.restricted_import,
        }

        # Global environment
        global_env = {"__builtins__": allowed_builtins}

        for i, var in enumerate(self.data):
            global_env[f"DF_{i+1}"] = var

        if st.session_state["geojson_var"] is not None:
            geojson = json.loads(st.session_state["geojson_var"])
        else:
            geojson = ""
        global_env["GEOJSON"] = [geojson]

        return global_env

    @abstractmethod
    def exec_code(self, code: str) -> str:
        """
        Executes provided code in a restricted environment.

        This method executes the provided Python code within a restricted environment. It then creates an isolated 
        execution environment using `create_isolated_env`, and executes the code within this environment. The result 
        of the executed code is retrieved from the context and returned.

        Args:
            code (str): The Python code to execute.

        Returns:
            str: The result of the executed code.
        """

        # Block plotly show() method
        go.Figure.show = self._intercept_show

        # Instanciate safe env
        env = self.create_isolated_env()
        context = {}
        exec(code, env, context)
        code_result = context["result"]

        return code_result

    @abstractmethod
    def _intercept_show(self):
        """
        Intercept and block Plotly's show() method.

        This method is used to intercept and block the execution of the Plotly `show()` method.
        If the `show()` method is called, it raises a RuntimeError indicating that the execution
        of the method is not allowed and should be removed from the code.

        Raises:
            RuntimeError: Indicates that the execution of the plotly `show()` method is not allowed.
        """
        raise RuntimeError(
            "Execution of the plotly fig.show() method is not allowed, remove it."
        )
