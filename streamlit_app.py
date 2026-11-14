import streamlit as st

# Define multi-page structure
lab1_page = st.Page("Lab1.py", title="Lab 1: Document QA")
lab2_page = st.Page("Lab2.py", title="Lab 2: Document Summarizer")
lab3_page = st.Page("Lab3.py", title="Lab 3: Streaming Chatbot")
lab4_page = st.Page("Lab4.py", title="Lab 4: RAG Chatbot")
lab5_page = st.Page("Lab5.py", title="Lab 5: What to Wear Bot", default=True)
lab8_page = st.Page("Lab 8/LabMultimodal.py", title="Lab 8: See and Hear Assistant")

# Navigation
pg = st.navigation([lab1_page, lab2_page, lab3_page, lab4_page, lab5_page, lab8_page])
pg.run()