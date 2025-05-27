import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
import openai

# Set up the model for embeddings
embedding_model = SentenceTransformer("nvidia/NV-Embed-v2")

# Configure OpenAI model API
openai.api_base = "http://localhost:1234/v1"
openai.api_key = "lm-studio"

# List to store the results
results = []

def get_response_from_model(query: str) -> str:
    try:
        completion = openai.ChatCompletion.create(
            model="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
            messages=[{"role": "system", "content": "You are InfoTechnology Bridge Assistant."},
                      {"role": "user", "content": query}],
            temperature=0.7,
        )
        return completion.choices[0].message['content'].strip()
    except Exception as e:
        print(f"Error connecting to LLM: {e}")
        return "Error: Unable to retrieve response from the model."

def calculate_similarity(expected_response: str, actual_response: str) -> float:
    # Encode both responses to get embeddings
    expected_embedding = embedding_model.encode([expected_response], convert_to_tensor=True)
    actual_embedding = embedding_model.encode([actual_response], convert_to_tensor=True)
    
    # Calculate cosine similarity
    similarity_score = util.pytorch_cos_sim(expected_embedding, actual_embedding).item()
    return similarity_score

def run_manual_test():
    print("Manual LLM Model Accuracy Test")
    print("Type 'exit' to stop testing.")
    
    while True:
        query = input("\nEnter your query: ")
        if query.lower() == "exit":
            break

        # Get model response
        actual_response = get_response_from_model(query)
        print(f"\nModel Response:\n{actual_response}")
        
        # Manually input expected response
        expected_response = input("\nEnter the expected response: ")

        # Calculate similarity score
        similarity_score = calculate_similarity(expected_response, actual_response)
        print(f"Similarity Score: {similarity_score:.2f}")

        # Store result
        results.append({
            "query": query,
            "expected_response": expected_response,
            "actual_response": actual_response,
            "similarity_score": similarity_score
        })
        
    # Save results to a report
    save_report()

def save_report():
    # Save the results to a JSON or CSV file
    with open("manual_accuracy_report.json", "r") as f:
        json.dump(results, f, indent=4)
    print("\nManual accuracy report saved to 'manual_accuracy_report.json'")


if __name__ == "__main__":
    run_manual_test()
