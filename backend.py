
import os
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

def extract_video_id(url):
    import re
    pattern = r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})"
    match = re.search(pattern, url)
    if not match:
        raise ValueError("Invalid YouTube URL")
    return match.group(1)

def get_transcript(video_url):
    video_id = extract_video_id(video_url)
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    text = " ".join([t["text"] for t in transcript])
    return text

def create_vector_store(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([text])
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

def get_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer ONLY from the context below.
        If the context is insufficient, say you don't know.

        {context}
        Question: {question}
        """.strip(),
        input_variables=['context', 'question']
    )
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt}
    )
    return chain

def answer_question(video_url, question):
    transcript = get_transcript(video_url)
    vectorstore = create_vector_store(transcript)
    qa_chain = get_qa_chain(vectorstore)
    result = qa_chain.run(question)
    return result
