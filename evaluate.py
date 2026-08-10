import os
import json
import sys
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, context_recall
from langchain_groq import ChatGroq
from ragas.llms import LangchainLLMWrapper
from rag_pipeline import RAGPipeline

def main():
    print("--- Starting RAG Evaluation ---")
    
    # 1. Check for API key
    if "GROQ_API_KEY" not in os.environ:
        print("Error: GROQ_API_KEY environment variable is missing!")
        print("Please set it before running the evaluation.")
        sys.exit(1)
        
    # 2. Load the RAG Pipeline and the chosen strategy (Hybrid Search)
    print("Setting up Hybrid Search Strategy...")
    pipeline = RAGPipeline()
    retriever = pipeline.strategy_3_hybrid_search()

    # 3. Load the simulated test data (Golden Dataset)
    print("Loading Golden Dataset...")
    with open("data/golden_dataset.json", "r") as f:
        golden_data = json.load(f)

    # 4. Prepare data lists for Ragas
    questions = []
    ground_truths = []
    contexts_list = []

    print("Retrieving contexts for each question...")
    for item in golden_data:
        question = item["question"]
        ground_truth = item["ground_truth"]
        
        # Use our Hybrid Search to find the best documents
        retrieved_docs = retriever.invoke(question)
        
        # Extract just the text content from the retrieved documents
        contexts = [doc.page_content for doc in retrieved_docs]
        
        questions.append(question)
        ground_truths.append(ground_truth)
        contexts_list.append(contexts)

    # 5. Format into a HuggingFace Dataset, which is what Ragas expects
    data = {
        "question": questions,
        "ground_truth": ground_truths, 
        "contexts": contexts_list
    }
    dataset = Dataset.from_dict(data)

    # 6. Set up the LLM as a Judge using Groq
    # We use llama-3.3-70b-versatile as requested
    print("Initializing Groq LLM as a Judge (llama-3.3-70b-versatile)...")
    groq_llm = ChatGroq(model_name="llama-3.3-70b-versatile")
    ragas_llm = LangchainLLMWrapper(groq_llm)

    # 7. Run the Evaluation
    print("Running Evaluation using Ragas (Context Precision & Context Recall)...")
    # This might take a few moments as it makes calls to the LLM
    result = evaluate(
        dataset=dataset,
        metrics=[context_precision, context_recall],
        llm=ragas_llm,
    )
    
    # 8. Print Results and check against thresholds
    print("\n--- Final Evaluation Results ---")
    print(result)

    # The result object acts like a dictionary
    recall_score = result.get("context_recall", 0)
    precision_score = result.get("context_precision", 0)
    
    print(f"\nContext Precision: {precision_score:.2f}")
    print(f"Context Recall: {recall_score:.2f}")

    # Fail the CI pipeline if Recall is too low
    if recall_score < 0.80:
        print("FAIL: Context Recall is below the 0.80 threshold! ❌")
        sys.exit(1)
    else:
        print("SUCCESS: Context Recall meets the threshold! ✅")
        sys.exit(0)

if __name__ == "__main__":
    main()
