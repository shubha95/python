import streamlit as st

st.set_page_config(page_title="HR Policy Assistant", page_icon="🤖")
st.title("HR Policy Assistant")
st.write("Welcome to your HR Assistant!")

question = st.chat_input("Ask an HR question")
if question:
    st.write("Your question:", question)