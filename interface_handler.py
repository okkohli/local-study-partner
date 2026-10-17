"""Streamlit interface for the Study Expert RAG.

Run from the project folder: python -m streamlit run main_interface.py

"""

# cd C:\Users\kohlh\OneDrive\Projects\RAG\local-study-partner\venv\Scripts\python.exe -m streamlit run main_interface.py


import os
import re
from pathlib import Path

import streamlit as st

from main import build_chain


@st.cache_resource
def get_chain():
    """Load the same RAG pipeline as the CLI once across Streamlit reruns."""
    # The existing retriever resolves materials and the database relative to cwd.
    os.chdir(Path(__file__).resolve().parent)

    return build_chain()


def format_answer(answer):
    """Normalize LaTeX math delimiters for Markdown, preserving code blocks."""
    parts = re.split(r"(```[\s\S]*?```|`[^`\n]*`)", answer)
    for index in range(0, len(parts), 2):
        parts[index] = re.sub(
            r"\\\[([\s\S]*?)\\\]",
            lambda match: "\n\n$$\n" + match.group(1).strip() + "\n$$\n\n",
            parts[index],
        )
        parts[index] = re.sub(
            r"\\\(([\s\S]*?)\\\)",
            lambda match: "$" + match.group(1).strip() + "$",
            parts[index],
        )
    return "".join(parts)


def main():
    st.set_page_config(page_title="Study Expert", layout="wide")
    st.title("Study Expert")
    st.caption("Ask questions about your PDF study materials.")

    if "rag_answer" not in st.session_state:
        st.session_state.rag_answer = ""
    if "rag_question" not in st.session_state:
        st.session_state.rag_question = ""

    input_column, output_column = st.columns(2)

    with input_column:
        st.subheader("Input")
        with st.form("question_form"):
            question = st.text_area("Enter your question:", height=400)
            submitted = st.form_submit_button("Ask Study Expert", type="primary")

    if submitted:
        question = question.strip()
        if not question:
            st.warning("Please enter a question first.")
        else:
            try:
                with st.spinner("Searching your study materials and generating an answer..."):
                    answer = get_chain().invoke(question)
                st.session_state.rag_answer = answer["answer"]
                st.session_state.rag_question = question
            except Exception as exc:
                st.error(
                    "Could not generate an answer. Make sure Ollama is running "
                    "and the llama3.2:3b and mxbai-embed-large models are installed."
                )
                with st.expander("Error details"):
                    st.code(str(exc))

    with output_column:
        st.subheader("Output")
        if st.session_state.rag_question:
            st.caption(f'Answer to:\n"{st.session_state.rag_question}"')
        with st.container(height=450, border=True):
            if st.session_state.rag_answer:
                st.markdown(format_answer(st.session_state.rag_answer))
            else:
                st.caption("Your answer will appear here.")


if __name__ == "__main__":
    main()
