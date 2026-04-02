"""import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load API key from .env file
load_dotenv()

# Initialize the LLM
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

# First test call
response = llm.invoke("What is a Non-Performing Asset in Indian banking? Answer in 2 sentences.")

print(response.content)"""



from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

data_dir = 'data'

documents = []
for filename in os.listdir(data_dir):
    if filename.endswith(".txt"):
        filepath = os.path.join(data_dir, filename)
        loader = TextLoader(filepath, encoding = 'utf-8')
        loaded_docs = loader.load()
        for doc in loaded_docs:
            doc.metadata["source"] = filename
        documents.extend(loaded_docs)
        print(f"Loaded : {filename}")

print(f"\Total page loaded: {len(documents)}")
# print(f"\nFirst page content length: {len(documents[0].page_content)}")
# print(f"First page preview: '{documents[0].page_content[:200]}'")

splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)

chunks =splitter.split_documents(documents)
print(f"Total chunks created: {len(chunks)}")

print(f"\n--- first chunk preview ---")
print(chunks[0].page_content[:500])


from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

embedding_model = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(
                    documents = chunks,
                    embedding = embedding_model,
                    #persist_directory = "chroma_db"
                    )

print(f"\nVector store created with {vectorstore._collection.count()} chunks")

'''query = "What are the fair practices for loan recovery by NBFCs?"
result = vectorstore.similarity_search(query, k=3)

print(f"\nQuery : {query}")
print(f"Top {len(result)} results:\n")

for i, doc in enumerate(result, 1):
    print(f"----Result {i}----")
    print(doc.page_content[:300])
    print()'''


from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


load_dotenv()

llm = ChatGroq(model_name = "llama-3.3-70b-versatile", temperature = 0)
retriever = vectorstore.as_retriever(search_kwargs = {"k" : 4})
prompt_template = ChatPromptTemplate.from_template(
                    """ You are an expert assistant on Indian Banking and NBFC regulations.
        Answer the question by ONLY using the context provided below.
        If the context does not contain enough information to answer, say:
        "The preovided regulatory documents do not contain sufficient information to answer the questions".

        For every claim in your answer, mention which document it comes from (use the source filename from the metadata).
        Context: {context}
        Question: {question}

        Answer:"""
)

def format_docs(docs):
    formatted = []
    for docs in docs:
        source = docs.metadata.get("source", "Unknown")
        formatted.append(f"[Source: {source}]\n {docs.page_content}")
    return "\n\n".join(formatted)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt_template | llm | StrOutputParser()
)

test_questions = [
   "What is the minimum capital adequacy ratio required for NBFC-MFIs?"
]

for q in test_questions:
    print(f"\n{'='*80}")
    print(f"QUESTION: {q}")
    print(f"{'='*80}")
    answer = rag_chain.invoke(q)
    print(f"\nANSWER:\n{answer}")