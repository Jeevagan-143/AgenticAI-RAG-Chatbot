import streamlit as st

from src.graph import graph


st.set_page_config(
    page_title="Agentic AI RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)


st.title("🤖 Agentic AI RAG Chatbot")

st.write(
    "Ask questions based on the Agentic AI eBook."
)

st.info(
    "This chatbot answers using information retrieved "
    "from the Agentic AI eBook."
)


query = st.text_input(
    "Ask a question",
    placeholder="What is Agentic AI?"
)


if st.button("Ask"):

    if not query.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner("Searching the Agentic AI eBook..."):

            result = graph.invoke({
                "question": query,
                "context": [],
                "answer": "",
                "score": 0.0
            })


        st.subheader("Answer")

        st.write(result["answer"])


        st.subheader("Confidence Score")

        st.write(
            f"{result['score']:.4f}"
        )


        st.subheader("Retrieved Sources")

        for chunk in result["context"]:

            with st.expander(
                f"Page {int(chunk['page'])}"
            ):

                st.write(chunk["text"])

                st.caption(
                    f"Source: {chunk['source']}"
                )