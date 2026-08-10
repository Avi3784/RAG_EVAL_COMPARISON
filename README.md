# RAG Evaluation Comparison Project

## What is this project?
This project builds a system called RAG (Retrieval-Augmented Generation). Think of RAG like giving an AI an open-book test. Instead of answering from memory, the AI searches a specific document to find the right answer.

I built this project to test four different ways of searching for information (called "retrieval strategies") to see which one works best. Then, I used another AI to automatically grade how well the search worked.

## How it works (Step by Step)

### 1. Preparing the Data
- **What it is:** We take a text document and break it into smaller pieces called "chunks".
- **Why we do it:** An AI cannot read a giant book all at once. We break it into small pieces so the AI can read just the relevant parts.
- **How it is done:** We use LangChain tools to split the text.

### 2. Embeddings and Vector Database
- **What it is:** We turn text chunks into numbers (vectors) and store them in a database.
- **Why we do it:** Computers understand numbers better than words. When text is turned into numbers, we can use math to find sentences that have similar meanings.
- **How it is done:** We use a free, local model called HuggingFace `BAAI/bge-small-en-v1.5` to turn text into numbers. We store these numbers in a database called ChromaDB.

### 3. The Four Search Strategies
I built four different ways to search the database.

**Strategy 1: Simple Chunking**
- **What:** Splits text into fixed sizes (like exactly 500 letters per chunk).
- **Why:** It is the easiest and most common way to start.
- **How:** Uses a tool called `RecursiveCharacterTextSplitter`. It searches for chunks that have similar meaning to the user's question.

**Strategy 2: Semantic Chunking**
- **What:** Splits text based on meaning instead of a fixed size. For example, it keeps a whole paragraph together if it talks about the same topic.
- **Why:** It prevents cutting a sentence in half, which can confuse the AI.
- **How:** Uses a tool called `SemanticChunker`.

**Strategy 3: Hybrid Search**
- **What:** Combines two types of search: meaning-based search (Vector Search) and exact word match search (BM25).
- **Why:** Sometimes you want to find the exact word a user typed, not just words with similar meaning.
- **How:** Uses `EnsembleRetriever` to run both searches and mix the results 50/50.

**Strategy 4: Re-ranking**
- **What:** Pulls a lot of results first (like 10), then uses a very smart, slow model to grade and rank them to find the top 2.
- **Why:** Vector databases are fast but sometimes make mistakes. A re-ranker is slow but very accurate. Using both gives you speed and accuracy.
- **How:** Uses a HuggingFace Cross-Encoder model (`ms-marco-MiniLM-L-6-v2`) to re-rank the results from ChromaDB.

### 4. Automated Evaluation (LLM-as-a-Judge)
- **What it is:** Using an AI model to grade another AI system.
- **Why we do it:** Checking answers manually takes too long. We want a fast, automated way to know if our search strategies are actually good.
- **How it is done:** I used a framework called Ragas and the Groq API (`llama-3.3-70b-versatile` model). The judge grades two things:
  - **Context Precision:** Did we find the right documents, and did we put the best one at the top?
  - **Context Recall:** Did we find all the information needed to answer the question, or did we miss something?

## Results Comparison

Here is how the four search strategies performed when graded by the AI judge.

| Retrieval Strategy | Context Precision | Context Recall | Notes |
| :--- | :--- | :--- | :--- |
| **Simple Chunking** | 0.70 | 0.65 | Fast, but misses context by cutting sentences in half. |
| **Semantic Chunking** | 0.75 | 0.75 | Better than simple chunking because it keeps related sentences together. |
| **Hybrid Search** | 0.85 | 0.85 | Big jump in performance. Exact word matching helps find specific terms. |
| **Re-ranking** | 0.95 | 1.00 | The best results. The re-ranker perfectly ordered the top documents. |

## CI/CD Pipeline (GitHub Actions)
- **What it is:** An automated script that runs every time I push new code.
- **Why we do it:** To make sure I do not accidentally break the system when I add new code.
- **How it is done:** I wrote a GitHub Actions workflow. Every time code is pushed, it runs the evaluation script. If the Context Recall score falls below 0.80, the pipeline fails and warns me that the search quality is too low.