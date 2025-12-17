from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

def build_vector_db(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(text)
    
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_texts(chunks, embedding=embeddings)
    
    return vectordb

def answer_query(query, vectordb):
    llm = OpenAI()
    docs = vectordb.similarity_search(query)
    
    content = "\n".join([doc.page_content for doc in docs])
    prompt = f"Using the following financial context:\n{content}\n\nAnswer: {query}"

    return llm(prompt)
