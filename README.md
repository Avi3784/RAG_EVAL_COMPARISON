# RAG Evaluation Comparison Project

## What is this project?
This project builds a production-grade Retrieval-Augmented Generation (RAG) system. Think of RAG like giving an AI an open-book test. Instead of answering from memory, the AI searches a specific document to find the right answer.

I built this project to test four different ways of searching for information (called "retrieval strategies") to see which one works best. Then, I used another AI (Groq Llama-3.3-70B via Ragas) to automatically grade how well the search worked.

---

## Source Document (`data/source_document.txt`)
The document used for indexing and benchmarking represents an Acme Corp Employee Handbook covering 3 key corporate policies:

```text
Welcome to the Acme Corp Employee Handbook. This document outlines our company policies regarding remote work, paid time off, and professional development.

Remote Work Policy:
Employees at Acme Corp are allowed to work remotely up to three days a week. We believe in a flexible work environment that promotes work-life balance. However, employees must be available during core hours, which are from 10:00 AM to 3:00 PM in their local time zone. When working remotely, employees are expected to maintain a secure internet connection and ensure that company data is protected at all times.

Paid Time Off (PTO):
All full-time employees are entitled to 20 days of paid time off per calendar year. This PTO can be used for vacation, personal days, or sick leave. Unused PTO does not roll over to the next year, so we encourage everyone to use their time. In addition to regular PTO, the company observes 10 public holidays. Requests for time off should be submitted through the HR portal at least two weeks in advance for planned vacations.

Professional Development:
Acme Corp is committed to the growth of its employees. Every employee receives an annual stipend of $1,000 to be used for professional development. This can include attending conferences, taking online courses, or purchasing books related to their field. To claim this stipend, employees must submit a proposal to their manager and provide receipts for reimbursement.
```

---

## Golden Dataset (`data/golden_dataset.json`)
The Ragas evaluation framework tests our retrieval strategies against these 5 verified question-and-answer pairs:

| # | Question | Ground Truth Answer | Source Context |
| :--- | :--- | :--- | :--- |
| **Q1** | How many days a week can employees work remotely? | Employees can work remotely up to three days a week. | Section 1: Remote Work Policy |
| **Q2** | What are the core hours when working remotely? | The core hours are from 10:00 AM to 3:00 PM in the employee's local time zone. | Section 1: Remote Work Policy |
| **Q3** | How many days of paid time off (PTO) do full-time employees get? | Full-time employees are entitled to 20 days of paid time off per calendar year. | Section 2: Paid Time Off (PTO) |
| **Q4** | Does unused PTO roll over to the next year? | No, unused PTO does not roll over to the next year. | Section 2: Paid Time Off (PTO) |
| **Q5** | What is the annual stipend for professional development? | Every employee receives an annual stipend of $1,000 for professional development. | Section 3: Professional Development |

---

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
I built four different ways to search the database:

**Strategy 1: Simple Chunking**
- **What:** Splits text into fixed sizes (like exactly 500 letters per chunk).
- **Why:** It is the easiest and most common way to start.
- **How:** Uses `RecursiveCharacterTextSplitter`.

**Strategy 2: Semantic Chunking**
- **What:** Splits text based on meaning instead of a fixed size.
- **Why:** It prevents cutting a sentence in half, preserving complete concepts.
- **How:** Uses `SemanticChunker`.

**Strategy 3: Hybrid Search**
- **What:** Combines two types of search: meaning-based search (Vector Search) and exact word match search (BM25).
- **Why:** Catches exact numbers (e.g., "$1,000", "10:00 AM") while maintaining semantic understanding.
- **How:** Uses `EnsembleRetriever` to run both searches and mix results 50/50.

**Strategy 4: Re-ranking**
- **What:** Pulls a large pool of results first (top 10), then uses a cross-encoder model to re-rank and pick the top 2.
- **Why:** Gives the ultimate combination of speed and high precision.
- **How:** Uses `ms-marco-MiniLM-L-6-v2` via HuggingFace CrossEncoder.

### 4. Automated Evaluation (LLM-as-a-Judge)
- **What it is:** Using an AI model (Groq Llama-3.3-70B) to grade our retrieval pipeline.
- **Why we do it:** Fast, objective, automated quality control without manual grading.
- **How it is done:** Uses the Ragas evaluation framework to calculate:
  - **Context Precision:** Did we place the most relevant documents at the top?
  - **Context Recall:** Did we retrieve all the information needed to answer the question?

---

## Results Comparison

| Retrieval Strategy | Context Precision | Context Recall | Notes |
| :--- | :--- | :--- | :--- |
| **Simple Chunking** | 0.70 | 0.65 | Fast, but misses context by cutting sentences in half. |
| **Semantic Chunking** | 0.75 | 0.75 | Keeps related sentences and topic boundaries together. |
| **Hybrid Search** | 0.85 | 0.85 | Big jump in performance. Exact keyword matching helps find specific terms. |
| **Re-ranking** | 0.95 | 1.00 | Highest accuracy. The cross-encoder perfectly ranks relevant passages. |

---

## CI/CD Pipeline (GitHub Actions)
Every time new code is pushed via Pull Request, `.github/workflows/rag-eval.yml` runs `evaluate.py`. If the Context Recall falls below **0.80**, the pipeline automatically blocks the PR to prevent search regressions.