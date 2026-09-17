import streamlit as st

# Define multi-page structure
lab1_page = st.Page("Lab1.py", title="Lab 1: Document QA")
lab2_page = st.Page("Lab2.py", title="Lab 2: Document Summarizer")
lab3_page = st.Page("Lab3.py", title="Lab 3: Streaming Chatbot")
lab4_page = st.Page("Lab4.py", title="Lab 4: RAG Chatbot", default=True)

# Navigation
pg = st.navigation([lab1_page, lab2_page, lab3_page, lab4_page])
pg.run()










# import streamlit as st
# from openai import OpenAI

# # Show title and description.
# st.title("My Document question answering")
# st.write(
#     "Upload a document below and ask a question about it – GPT will answer! "
#     "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
# )

# # Ask user for their OpenAI API key via `st.text_input`.
# # Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# # via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
# openai_api_key = st.text_input("OpenAI API Key", type="password")
# if not openai_api_key:
#     st.info("Please add your OpenAI API key to continue.", icon="🗝️")
# else:
#     # Create an OpenAI client
#     client = OpenAI(api_key=openai_api_key)

#     # Validate the API key immediately
#     try:
#         client.models.list()
#     except Exception as e:
#         st.error("Invalid OpenAI API Key. Please check your key and try again.")
#         st.stop()

#     # Let the user upload a file via `st.file_uploader`
#     uploaded_file = st.file_uploader(
#         "Upload a document (.txt or .md)", type=("txt", "md")
#     )

#     # Ask the user for a question via `st.text_area`
#     question = st.text_area(
#         "Now ask a question about the document!",
#         placeholder="Can you give me a short summary?",
#         disabled=not uploaded_file,
#     )

#     if uploaded_file and question:
#         # Process the uploaded file and question
#         document = uploaded_file.read().decode()
#         messages = [
#             {
#                 "role": "user",
#                 "content": f"Here's a document: {document} \n\n---\n\n {question}",
#             }
#         ]

#         # Generate an answer using the OpenAI API
#         stream = client.chat.completions.create(
#             model="gpt-4.1",
#             messages=messages,
#             stream=True,
#         )

#         # Stream the response to the app using `st.write_stream`
#         st.write_stream(stream)
