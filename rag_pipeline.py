import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceBgeEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever

class RAGPipeline:
    def __init__(self, data_path="data/source_document.txt"):
        """
        Initialize the RAG Pipeline.
        This loads our text file and sets up the local HuggingFace embeddings.
        """
        self.data_path = data_path
        
        # Load the document using a basic TextLoader
        print(f"Loading document from {self.data_path}...")
        loader = TextLoader(self.data_path)
        self.docs = loader.load()
        
        # Initialize the Embeddings model
        # We are using a small BGE model which is great for local, free embeddings!
        model_name = "BAAI/bge-small-en-v1.5"
        print(f"Initializing Embeddings: {model_name}...")
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},  # Run on CPU for broad compatibility
            encode_kwargs={'normalize_embeddings': True}
        )

    def strategy_1_simple_chunking(self):
        """
        Strategy 1: Simple Chunking
        Splits the text into simple, fixed-size chunks.
        """
        print("\n--- Running Strategy 1: Simple Chunking ---")
        # Split text into chunks of 500 characters, with 50 characters overlap
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = text_splitter.split_documents(self.docs)
        
        # Store these chunks in a Chroma Vector Database
        vectorstore = Chroma.from_documents(chunks, self.embeddings)
        
        # Return a retriever that fetches the top 3 similar chunks
        return vectorstore.as_retriever(search_kwargs={"k": 3})

    def strategy_2_semantic_chunking(self):
        """
        Strategy 2: Semantic Chunking
        Splits the text based on meaning (semantics) rather than fixed sizes.
        """
        print("\n--- Running Strategy 2: Semantic Chunking ---")
        # This splitter groups sentences that are semantically similar
        text_splitter = SemanticChunker(self.embeddings)
        chunks = text_splitter.split_documents(self.docs)
        
        # Store these chunks in a Chroma Vector Database
        vectorstore = Chroma.from_documents(chunks, self.embeddings)
        
        # Return a retriever that fetches the top 3 similar chunks
        return vectorstore.as_retriever(search_kwargs={"k": 3})

    def strategy_3_hybrid_search(self):
        """
        Strategy 3: Hybrid Search
        Combines Vector Search (understanding meaning) with BM25 (exact keyword matching).
        """
        print("\n--- Running Strategy 3: Hybrid Search ---")
        # First, split the text simply
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(self.docs)
        
        # 1. Setup Vector Retriever (Semantic search)
        vectorstore = Chroma.from_documents(chunks, self.embeddings)
        vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # 2. Setup BM25 Retriever (Keyword search)
        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = 3
        
        # 3. Combine both retrievers using an EnsembleRetriever
        # It weights both results equally (0.5 and 0.5)
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.5, 0.5]
        )
        return ensemble_retriever

    def strategy_4_reranking(self):
        """
        Strategy 4: Re-ranking
        Fetches a large pool of results (top 10), then uses a cross-encoder model 
        to precisely score and pick the absolute best (top 2).
        """
        print("\n--- Running Strategy 4: Re-ranking ---")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(self.docs)
        
        # Store chunks in Chroma, but fetch Top 10 initially!
        vectorstore = Chroma.from_documents(chunks, self.embeddings)
        base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
        
        # Initialize the HuggingFace CrossEncoder for re-ranking
        model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        model = HuggingFaceCrossEncoder(model_name=model_name)
        
        # Setup the reranker to keep only the top 2 results
        compressor = CrossEncoderReranker(model=model, top_n=2)
        
        # Combine the base retriever and the compressor
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
        return compression_retriever

# Quick test if you run this file directly
if __name__ == "__main__":
    pipeline = RAGPipeline()
    retriever = pipeline.strategy_1_simple_chunking()
    results = retriever.invoke("What is the remote work policy?")
    print(f"Found {len(results)} chunks for simple chunking.")
    for res in results:
        print("-", res.page_content[:100], "...")
