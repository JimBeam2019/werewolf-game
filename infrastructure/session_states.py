import streamlit as st


def initialize_params():
    """
    Function for initializing the session states
    """

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
