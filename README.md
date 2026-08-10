# RAG Evaluation Pipeline

This repository contains a production-grade Retrieval-Augmented Generation (RAG) pipeline designed to systematically compare various document retrieval strategies. To ensure high-quality outputs, the system utilizes an automated "LLM-as-a-Judge" evaluation framework using the Ragas library and the Groq API (llama-3.3-70b-versatile).

## System Architecture

The architecture is divided into three primary components: Data Processing, Retrieval Strategies, and Automated Evaluation.

```mermaid
graph TD
    classDef file fill:#f9f2f4,stroke:#c7254e,stroke-width:2px,color:#c7254e
    classDef process fill:#e1f5fe,stroke:#03a9f4,stroke-width:2px,color:#01579b
    classDef storage fill:#e8f5e9,stroke:#4caf50,stroke-width:2px,color:#1b5e20
    classDef llm fill:#fff3e0,stroke:#ff9800,stroke-width:2px,color:#e65100
    classDef decision fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px,color:#4a148c
    
    subgraph Data Ingestion and Indexing
        A[source_document.txt]:::file -->|LangChain TextLoader| B(Raw Text Document):::process
        B -->|RecursiveCharacterTextSplitter| C1(Simple Chunks):::process
        B -->|SemanticChunker| C2(Semantic Chunks):::process
        
        C1 -->|HuggingFace: BAAI/bge-small| D(Chroma Vector DB):::storage
        C2 -->|HuggingFace: BAAI/bge-small| D
    end

    subgraph Retrieval Strategies
        D -->|Vector Search k=3| S1(Strategy 1: Simple)
        D -->|Vector Search k=3| S2(Strategy 2: Semantic)
        
        D -->|Vector Search k=3| H1(Semantic Search)
        C1 -->|BM25 Retriever k=3| H2(Keyword Search)
        H1 -->|EnsembleRetriever 50/50| S3(Strategy 3: Hybrid Search):::process
        H2 -->|EnsembleRetriever 50/50| S3
        
        D -->|Vector Search k=10| R1(Base Retrieval)
        R1 -->|ms-marco-MiniLM-L-6-v2 top=2| S4(Strategy 4: Re-ranking)
    end

    subgraph Automated Evaluation Framework
        G[golden_dataset.json]:::file -->|Load Q and A Pairs| E1(Extract Questions):::process
        E1 -->|Query| S3
        S3 -->|Retrieved Contexts| E2(Data Preparation):::process
        E2 -->|questions, contexts, ground_truth| E3(HuggingFace Dataset):::storage
        
        E3 --> Ragas[Ragas Evaluator]:::process
        LLM[Groq API: llama-3.3-70b-versatile]:::llm -->|LangchainLLMWrapper| Ragas
        
        Ragas -->|Context Precision and Recall| Dec{Is Recall >= 0.80?}:::decision
        Dec -->|Yes| Pass[Exit 0: GitHub Actions PASS]:::storage
        Dec -->|No| Fail[Exit 1: GitHub Actions FAIL]:::file
    end
```

## Retrieval Strategies
1. **Simple Retrieval**: Standard semantic search utilizing a fixed-size character splitter.
2. **Semantic Retrieval**: Advanced chunking utilizing sentence embeddings to group text by semantic meaning.
3. **Hybrid Search**: Combines dense vector search (ChromaDB) with sparse keyword search (BM25) via LangChain's EnsembleRetriever.
4. **Re-ranking**: Retrieves a broad candidate pool from ChromaDB and applies a HuggingFace Cross-Encoder (ms-marco-MiniLM-L-6-v2) for precise scoring of the top results.

## Evaluation and Test Results

The pipeline evaluates the Hybrid Search strategy against a predefined golden dataset utilizing the Ragas framework. The Groq API (llama-3.3-70b-versatile) serves as the impartial judge.

### Execution Output

```text
--- Starting RAG Evaluation ---
Setting up Hybrid Search Strategy...
Loading document from data/source_document.txt...
Initializing Embeddings: BAAI/bge-small-en-v1.5...
Loading Golden Dataset...
Retrieving contexts for each question...
Initializing Groq LLM as a Judge (llama-3.3-70b-versatile)...
Running Evaluation using Ragas (Context Precision & Context Recall)...

--- Final Evaluation Results ---
{'context_precision': 0.9500, 'context_recall': 1.0000}

Context Precision: 0.95
Context Recall: 1.00
SUCCESS: Context Recall meets the threshold.
```

### Continuous Integration
A GitHub Actions workflow is configured to execute this evaluation automatically on every pull request. If the Context Recall score falls below the required threshold of 0.80, the pipeline will fail, preventing the integration of degraded retrieval logic.