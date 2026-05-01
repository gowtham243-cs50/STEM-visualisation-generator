# build_manim_index.py
from pydantic import SecretStr

import os
from dotenv import load_dotenv
os.environ["USER_AGENT"] = "manim-generator/1.0"
os.environ["GOOGLE_API_KEY"] = "AIzaSyD7s1amUzk43rfQyDuGQw1ZmB1jkG2vNNk"
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()
# 1) Load docs from Manim docs site (extend this list as needed)
urls = [
    "https://manim.readthedocs.io/en/latest/",
    "https://manim.readthedocs.io/en/latest/reference/manim.mobject.text.tex_mobject.html",
    "https://manim.readthedocs.io/en/latest/reference/manim.mobject.graphing.axes.html",
]
loader = WebBaseLoader(urls)
docs = loader.load()  # List[Document][web:116][web:119]
print("docs:", len(docs))

# 2) Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
)
chunks = splitter.split_documents(docs)
print("chunks:", len(chunks))

# Filter out empty or whitespace-only chunks
chunks = [chunk for chunk in chunks if chunk.page_content.strip()]

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001",api_key=SecretStr(secret_value=os.getenv("GEMINI_API_KEY")or" "))    

# 4) Create / persist Chroma vector store
persist_dir = "./manim_chroma"
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_dir,
)
print("Built Manim index at", persist_dir)  