import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    formatted = []
    for docs in docs:
        source = docs.metadata.get("source", "Unknown")
        formatted.append(f"[Source: {source}]\n {docs.page_content}")
    return "\n\n".join(formatted)

def build_rag_chain(data_dir="data"):
    load_dotenv()

    if not os.environ.get('GROQ_API_KEY'):
        try:
            import streamlit as st
            os.environ['GROQ_API_KEY'] = st.secrets['GROQ_API_KEY']
        except(ImportError, KeyError):
            raise ValueError('GROQ_API_KEY not found in .env or Streamlit secrets')

    # Load documents
    documents = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(data_dir, filename)
            loader = TextLoader(filepath, encoding="utf-8")
            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.metadata["source"] = filename
            documents.extend(loaded_docs)

    # Chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    # Embed and store
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embedding_model)

    # Build chain
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt_template = ChatPromptTemplate.from_template(
        """You are an expert assistant on Indian Banking and NBFC regulations.
Answer the question by ONLY using the context provided below.
If the context does not contain enough information to answer, say:
"The provided regulatory documents do not contain sufficient information to answer this question."

For every claim in your answer, mention which document it comes from (use the source filename from the metadata).

Context: {context}

Question: {question}

Answer:"""
    )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return rag_chain, len(chunks)