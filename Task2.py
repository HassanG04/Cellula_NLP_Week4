import os
from typing import List
from dotenv import load_dotenv

import openai
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate


load_dotenv("C:/Users/ADMIN/Documents/LLMS/.env")

key = os.getenv("OPENROUTER_API_KEY")
if not key:
    raise ValueError("OPENROUTER_API_KEY not found in .env")

openai.api_key = key
openai.base_url = "https://openrouter.ai/api/v1"


class ChromaRAGSystem:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory

        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=key,
            openai_api_base="https://openrouter.ai/api/v1"
        )

        # Initialize LLM
        self.llm = ChatOpenAI(
            model="anthropic/claude-3.5-sonnet",
            temperature=0.7,
            openai_api_key=key,
            openai_api_base="https://openrouter.ai/api/v1"
        )

        # Initialize vector store
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )

    def add_documents(self, texts: List[str], metadatas: List[dict] = None):
        """Add documents to the vector store."""
        documents = [Document(page_content=text) for text in texts]
        if metadatas:
            for doc, meta in zip(documents, metadatas):
                doc.metadata = meta
        self.vectorstore.add_documents(documents)

    def add_documents_from_file(self, file_path: str, chunk_size: int = 1000):
        """Add documents from a text file with chunking."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=200
        )
        chunks = splitter.split_text(text)
        metadatas = [{"source": file_path, "chunk": i} for i in range(len(chunks))]
        self.add_documents(chunks, metadatas)

    def search(self, query: str, k: int = 3):
        """Search for similar documents."""
        return self.vectorstore.similarity_search(query, k=k)

    def ask(self, question: str):
        """Ask a question using RAG."""
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        prompt = ChatPromptTemplate.from_template(
            """You are a helpful AI assistant. Use the following context to answer the question. 
            If you don't know the answer based on the context, say so.

            Context:
            {context}

            Question: {question}

            Answer:"""
        )

        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
        )

        response = chain.invoke(question)
        return response.content

    def get_stats(self):
        """Get statistics about the vector store."""
        return {
            "total_documents": self.vectorstore._collection.count(),
            "persist_directory": self.persist_directory
        }

    def clear_database(self):
        """Clear all documents from the vector store."""
        self.vectorstore.delete_collection()
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )


def demo_basic_usage():
    """Demo function to test the RAG system."""
    rag = ChromaRAGSystem(persist_directory="./demo_chroma_db")

    # Sample documents
    documents = [
        "Python is a high-level programming language known for its simplicity and readability.",
        "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.",
        "Vector databases store data as high-dimensional vectors, which are perfect for similarity search.",
        "LangChain is a framework for building applications powered by large language models.",
        "Embeddings are numerical representations of text that capture semantic meaning."
    ]

    # Add documents
    rag.add_documents(documents)

    print("=== Search Demo ===")
    print("\nSearch Results for 'What is Python?':")
    results = rag.search("What is Python?", k=2)
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.page_content}")

    print("\n=== Q&A Demo ===")
    print("Question: What is machine learning?")
    answer = rag.ask("What is machine learning?")
    print(f"Answer: {answer}")

    print("\n=== Statistics ===")
    stats = rag.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")


def demo_file_ingestion():
    """Demo for ingesting documents from a file."""
    rag = ChromaRAGSystem(persist_directory="./file_demo_db")
    
    # Create a sample file
    sample_content = """Artificial Intelligence (AI) is the simulation of human intelligence in machines.
    Machine Learning is a subset of AI that focuses on algorithms that can learn from data.
    Deep Learning is a subset of machine learning using neural networks with multiple layers.
    Natural Language Processing enables computers to understand and process human language."""
    
    with open("sample_ai_concepts.txt", "w", encoding="utf-8") as f:
        f.write(sample_content)
    
    # Ingest the file
    rag.add_documents_from_file("sample_ai_concepts.txt")
    
    print("=== File Ingestion Demo ===")
    answer = rag.ask("What is the relationship between AI and machine learning?")
    print(f"Answer: {answer}")
    
    # Clean up
    if os.path.exists("sample_ai_concepts.txt"):
        os.remove("sample_ai_concepts.txt")


if __name__ == "__main__":
    demo_basic_usage()
    print("\n" + "="*50 + "\n")
    demo_file_ingestion()