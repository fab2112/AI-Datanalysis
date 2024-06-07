# Style Streamlit
import streamlit as st


def process_styles() -> None:
    """
    Customize main page styles.
    """

    # Text font - CSS File
    with open("css/style.css") as css:
        st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

    # Hide default about
    st.markdown(
        """
            <style>
                .st-emotion-cache-15yi2hn div[data-testid="stMarkdownContainer"] p:nth-of-type(2) {
                display: none;}
            </style>
        """,
        unsafe_allow_html=True,
    )

    # Position user chat on the left 
    st.markdown(
        """
            <style>
                .st-emotion-cache-janbn0 {
                    flex-direction: row-reverse;
                    text-align: left;
                    margin-left: 25px;
                }
                .st-emotion-cache-1c7y2kd {
                    flex-direction: row-reverse;
                    text-align: left;
                    margin-left: 25px;
                }
                
            </style>
        """,
        unsafe_allow_html=True,
    )

    # SideBar title | page
    st.markdown(
        """
            <style>
                .e1nzilvr2 {margin-top: -70px;}
                .e1nzilvr1 {margin-top: 50px;}
                .ea3mdgi5 {margin-top: -150px;}
            </style>
        """,
        unsafe_allow_html=True,
    )

    # ChatInput
    st.markdown(
        """
            <style>
                .stChatInput {width: 100%;}
                .st-emotion-cache-s1k4sy {width: 100%;}
            </style>
        """,
        unsafe_allow_html=True,
    )

    # Menu | header | footer
    st.markdown(
        """
            <style>
                .stDeployButton {visibility: hidden;}
                #MainMenu {visibility: visible;}
                footer {visibility: hidden;}
                header {visibility: hidden; }
                .stDeployButton {visibility: hidden;}
                .st-emotion-cache-cnbvxy e1nzilvr5 {display: none !important;}
            </style>
        """,
        unsafe_allow_html=True,
    )
    


