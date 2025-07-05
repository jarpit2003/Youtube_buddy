
import streamlit as st
from backend import answer_question

st.set_page_config(page_title="YouTube Video QA", layout="centered")
st.title("🎥 Ask AI about any YouTube Video")

video_url = st.text_input("Enter YouTube Video URL:")
question = st.text_input("Ask a question about the video:")

if st.button("Get Answer"):
    if not video_url or not question:
        st.warning("Please enter both a video URL and a question.")
    else:
        with st.spinner("Processing..."):
            try:
                answer = answer_question(video_url, question)
                st.success("Answer:")
                st.write(answer)
            except Exception as e:
                st.error(f"Error: {str(e)}")

st.markdown("---")
st.caption("Built with LangChain, FAISS, and Google Generative AI")
