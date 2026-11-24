import streamlit as st

from src.rag import get_rag_chain

st.set_page_config(page_title="HR Policy Assistant", page_icon=":material/support_agent:")
st.title("HR Policy Assistant")
st.caption("Ask questions about the company HR policy. Answers come only from the policy document.")


@st.cache_resource(show_spinner="Loading HR policy index...")
def load_chain():
    return get_rag_chain()


if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("Chat")
    if st.button("Clear conversation", icon=":material/delete:", width="stretch"):
        st.session_state.messages = []
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if question := st.chat_input("Ask an HR question", submit_mode="disable"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        try:
            chain = load_chain()
            answer = st.write_stream(chain.stream(question))
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as exc:
            st.error(
                f"Could not get an answer: {exc}",
                icon=":material/error:",
            )
