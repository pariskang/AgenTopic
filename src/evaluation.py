import re
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
from litellm import completion
import src.config as config

def evaluate_model(iteration, embedding_model, documents, topics):
    # Get embeddings for evaluation
    embeddings = embedding_model.embed_documents(documents)

    # Reduce dimensions for evaluation metrics
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    # Map topics to integers
    unique_topics = list(set(topics))
    topic_int_map = {topic: idx for idx, topic in enumerate(unique_topics)}
    topics_int = [topic_int_map[topic] for topic in topics]

    # Compute evaluation metrics
    silhouette_avg = silhouette_score(embeddings_2d, topics_int)
    calinski_harabasz = calinski_harabasz_score(embeddings_2d, topics_int)
    davies_bouldin = davies_bouldin_score(embeddings_2d, topics_int)

    print(f"Silhouette Score: {silhouette_avg}")
    print(f"Calinski-Harabasz Index: {calinski_harabasz}")
    print(f"Davies-Bouldin Index: {davies_bouldin}")

    # Store metrics
    return {
        'iteration': iteration,
        'silhouette_score': silhouette_avg,
        'calinski_harabasz_index': calinski_harabasz,
        'davies_bouldin_index': davies_bouldin
    }

def select_optimal_model(model_data):
    models = model_data['models']
    metrics = model_data['metrics']
    topic_summaries_list = model_data['topic_summaries_list']

    # Prepare a summary of metrics and topics for GPT-4
    evaluation_summary = "Here are the evaluation metrics for each iteration:\n"

    for metric in metrics:
        iteration = metric['iteration']
        evaluation_summary += f"\nIteration {iteration}:\n"
        evaluation_summary += f"- Silhouette Score: {metric['silhouette_score']}\n"
        evaluation_summary += f"- Calinski-Harabasz Index: {metric['calinski_harabasz_index']}\n"
        evaluation_summary += f"- Davies-Bouldin Index: {metric['davies_bouldin_index']}\n"
        # Add topic summaries
        evaluation_summary += "Topic Summaries:\n"
        topic_summaries = topic_summaries_list[iteration - 1]
        for topic_id, summary in topic_summaries.items():
            evaluation_summary += f"Topic {topic_id}: {summary}\n"

    # Create prompt for GPT-4 to select the best model
    prompt = evaluation_summary + """
Based on the evaluation metrics and the topic summaries for three iterations, please help identify the optimal iteration for topic modeling. Explain your reasoning and indicate which iteration (1, 2, or 3) provides the best balance between coherence and distinctiveness of topics.

Your response should be in the following format:

- Optimal Iteration: [Iteration Number]
- Reasoning: [Your explanation]
"""

    # Call GPT-4 to get the final advice
    response = completion(
        model=config.CHATGPT_MODEL_NAME,
        messages=[{"content": prompt, "role": "user"}]
    )
    advice = response['choices'][0]['message']['content'].strip()
    print("\nGPT-4's advice:")
    print(advice)

    # Parse GPT-4's recommendation
    optimal_iteration = None
    optimal_match = re.search(r'Optimal Iteration:\s*(\d+)', advice)
    if optimal_match:
        optimal_iteration = int(optimal_match.group(1))
        print(f"Selected optimal iteration: {optimal_iteration}")
    else:
        print("Could not determine optimal iteration from GPT-4's response.")
        optimal_iteration = len(models)  # Default to last iteration

    # Retrieve the optimal model
    optimal_model_data = next(
        (item for item in models if item['iteration'] == optimal_iteration),
        models[-1]
    )
    return optimal_model_data
        
