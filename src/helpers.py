from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
import os
from langchain.schema import Document
# ...existing code...


# ...existing code...
def load_pdf_file(data):
    loader = DirectoryLoader(
        data,
        glob='*.pdf',
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
  """
  Given a list of Document objects , a new list of Document objects 
  containing only 'source' in metadata and the original page_content.
  """
  minimal_docs:List[Document] = []
  for doc in docs:
    src = doc.metadata.get('source')
    minimal_docs.append(
      Document(
        page_content=doc.page_content,
        metadata={'source':src}
      )
    )
  return minimal_docs 

# Split the documentation into smaller chunks 
def text_splitter(minimal_docs):
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=500,
      chunk_overlap=20,
  )
  text_chunks = text_splitter.split_documents(minimal_docs)
  return text_chunks



#Download the Embedding from HuggingFace

def download_hugging_face_embedding():
  embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
  return embedding