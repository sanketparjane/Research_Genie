import os
from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

class RAGManager:
    def __init__(self, persist_dir: str = "./data/faiss_index"):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.persist_dir = persist_dir
        self.vector_store = None
        
        if os.path.exists(os.path.join(self.persist_dir, "index.faiss")):
            self.vector_store = FAISS.load_local(
                self.persist_dir, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )

    def add_texts(self, text: str, metadata: Dict[str, Any]):
        """
        Chunk the text, create embeddings, and add to FAISS vector store.
        """
        chunks = self.text_splitter.split_text(text)
        documents = [Document(page_content=chunk, metadata=metadata) for chunk in chunks]
        
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            self.vector_store.add_documents(documents)
            
        # Ensure directory exists before saving
        if not os.path.exists(self.persist_dir):
            os.makedirs(self.persist_dir)
            
        self.vector_store.save_local(self.persist_dir)

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """
        Retrieve top k most relevant chunks for a query.
        """
        if self.vector_store is None:
            return []
        return self.vector_store.similarity_search(query, k=k)
