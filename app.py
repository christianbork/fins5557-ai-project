"""FINS5557 Applied AI project - Streamlit entry point.

Run locally:  streamlit run app.py
"""
import streamlit as st

from src.config import GOOGLE_API_KEY

st.set_page_config(page_title="FINS5557 AI Project", layout="wide")
st.title("FINS5557 Applied AI Project")
st.write("Starter app. Replace this page once the team confirms the project idea.")

if GOOGLE_API_KEY:
    st.success("Gemini API key found.")
else:
    st.info("No GOOGLE_API_KEY set. Features that call Gemini will use mock output.")
