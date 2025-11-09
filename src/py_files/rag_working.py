
import os
from dotenv import load_dotenv
load_dotenv(override=True)

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool


pdf_folder = r"C:\AgenticAI\Projects_AgentAI\AgenticAI_04_companypolicy\pdf files"      # where pdf files are stored

pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith(".pdf")]          # collect all PDF files
print(f"len of pdf files {len(pdf_files)}.")


docs = []
for file in pdf_files:
    loader = PyPDFLoader(file)
    docs.extend(loader.load())
print(f"total pages loaded {len(docs)}")


text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs_splitter = text_splitter.split_documents(docs)
print(f"total documents chunked into docs split {len(docs_splitter)}")


embeddings = OpenAIEmbeddings()

## add these text to vectordb
vectorstore = FAISS.from_documents(
    documents = docs_splitter,
    embedding = embeddings
)

## convert vectordb to retreiver
retriever = vectorstore.as_retriever(search_kwargs={"k":3})

## retriever tool
retriever_resume_tool = create_retriever_tool(
    retriever,
    name="policy_validate_tool",
    description="searchSearch company policy documents to answer HR-related questions."
)

