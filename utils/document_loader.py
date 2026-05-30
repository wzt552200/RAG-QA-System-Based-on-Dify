import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_documents(directory: str):
    docs = []
    for file in os.listdir(directory):
        path = os.path.join(directory, file)
        if file.endswith(".txt"):
            loader = TextLoader(path, encoding="utf-8")
        elif file.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif file.endswith(".md"):
            loader = UnstructuredMarkdownLoader(path)
        else:
            continue
        docs.extend(loader.load())
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    return chunks