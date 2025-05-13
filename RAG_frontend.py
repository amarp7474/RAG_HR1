
import streamlit as st
import rag_backend as demo

st.set_page_config(page_title="HR QA and answers using RAG")
st.markdown("<h1 style='text-align: center;'>HR Q&A with RAG</h1>", unsafe_allow_html=True)

if 'vector_index' not in st.session_state:
    with st.spinner('Loading documents and building index...'):
        st.session_state.vector_index = demo.hr_index()

input_text = st.text_area('Ask your question here:', label_visibility='collapsed')
go_button = st.button("Submit")

if go_button:
    if not input_text.strip():
        st.warning("Please enter a question before submitting.")
    else:
        with st.spinner("Thinking..."):
            response = demo.hr_rag_response(index=st.session_state.vector_index, question=input_text)
            st.write(response)
