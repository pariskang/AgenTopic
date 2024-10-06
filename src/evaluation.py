# src/evaluation.py

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    v_measure_score,
    completeness_score,
    adjusted_rand_score,
    homogeneity_score
)
import re
from litellm import completion
import src.config as config

def compute_custom_ari(ari_original, n_c, n_t):
    """
    Compute the Custom Adjusted Rand Index (Custom ARI) based on the provided formula.

    Args:
        ari_original (float): The original Adjusted Rand Index.
        n_c (int): Number of predicted clusters.
        n_t (int): Number of true clusters.

    Returns:
        float: The Custom Adjusted Rand Index.
    """
    if n_t == 0 or min(n_c, n_t) == 0:
        print("Cannot compute Custom ARI due to zero clusters.")
        return np.nan

    beta = n_c / n_t
    gamma = max(n_c, n_t) / min(n_c, n_t)
    sign_ari = np.sign(ari_original)

    try:
        custom_ari = sign_ari * (beta / (1 - abs(ari_original))) * np.log(1 + abs(ari_original) * gamma)
    except ZeroDivisionError:
        custom_ari = np.inf if sign_ari > 0 else -np.inf
        print("Encountered division by zero in Custom ARI calculation. Assigned infinity based on sign.")
    
    return custom_ari

def evaluate_model(iteration, topic_model, embedding_model, test_documents, test_topics, true_labels):
    """
    Evaluate the topic model using various clustering metrics on the test dataset,
    including only the Custom Adjusted Rand Index.

    Args:
        iteration (int): Current iteration number.
        topic_model (BERTopic): Trained BERTopic model.
        embedding_model (SentenceTransformer): Embedding model instance.
        test_documents (list): List of test document texts.
        test_topics (list): List of assigned topic IDs for test documents.
        true_labels (list of str): List of true single labels for test documents.

    Returns:
        dict: Dictionary of evaluation metrics.
    """
    # Generate embeddings for test documents (if needed)
    embeddings = embedding_model.embed_documents(
        test_documents,
        verbose=True
    )

    # Reduce dimensions for evaluation metrics
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    # Ensure topic labels are strings for compatibility with scikit-learn metrics
    test_topics_str = [str(t) for t in test_topics]

    # Compute standard evaluation metrics
    silhouette_avg = silhouette_score(embeddings_2d, test_topics_str)
    calinski_harabasz = calinski_harabasz_score(embeddings_2d, test_topics_str)
    davies_bouldin = davies_bouldin_score(embeddings_2d, test_topics_str)
    
    # Primary labels are already single labels
    primary_labels = true_labels

    # Compute V-measure, Completeness, Homogeneity
    v_measure = v_measure_score(primary_labels, test_topics_str)
    completeness = completeness_score(primary_labels, test_topics_str)
    homogeneity = homogeneity_score(primary_labels, test_topics_str)

    # Compute Adjusted Rand Index (ARI)
    ari_original = adjusted_rand_score(primary_labels, test_topics_str)

    # Compute number of predicted and true clusters
    n_c = len(set(test_topics_str)) - (1 if '-1' in test_topics_str else 0)  # Exclude outliers
    n_t = len(set(primary_labels))

    # Compute Custom ARI
    custom_ari = compute_custom_ari(ari_original, n_c, n_t)

    # Compute Average Jaccard Similarity is skipped as per latest instructions

    print(f"\nIteration {iteration} Evaluation Metrics:")
    print(f"Silhouette Score: {silhouette_avg}")
    print(f"Calinski-Harabasz Index: {calinski_harabasz}")
    print(f"Davies-Bouldin Index: {davies_bouldin}")
    print(f"V-measure: {v_measure}")
    print(f"Completeness: {completeness}")
    print(f"Homogeneity: {homogeneity}")
    print(f"Custom Adjusted Rand Index (Custom ARI): {custom_ari}")

    # Store metrics
    return {
        'iteration': iteration,
        'silhouette_score': silhouette_avg,
        'calinski_harabasz_index': calinski_harabasz,
        'davies_bouldin_index': davies_bouldin,
        'v_measure': v_measure,
        'completeness': completeness,
        'homogeneity': homogeneity,
        'custom_adjusted_rand_index': custom_ari
    }

def select_optimal_model(model_data):
    """
    Select the optimal model based on evaluation metrics and GPT-4 feedback.

    Args:
        model_data (dict): Dictionary containing models, metrics, and topic summaries.

    Returns:
        dict: Optimal model data.
    """
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
        evaluation_summary += f"- V-measure: {metric['v_measure']}\n"
        evaluation_summary += f"- Completeness: {metric['completeness']}\n"
        evaluation_summary += f"- Homogeneity: {metric['homogeneity']}\n"
        evaluation_summary += f"- Custom Adjusted Rand Index (Custom ARI): {metric['custom_adjusted_rand_index']}\n"
        # Add topic summaries
        evaluation_summary += "Topic Summaries:\n"
        topic_summaries = topic_summaries_list[iteration - 1]
        for topic_id, summary in topic_summaries.items():
            evaluation_summary += f"Topic {topic_id}: {summary}\n"

    # Create prompt for GPT-4 to select the best model
    prompt = evaluation_summary + """
Based on the evaluation metrics and the topic summaries for all iterations, please help identify the optimal iteration for topic modeling. Explain your reasoning and indicate which iteration provides the best balance between coherence and distinctiveness of topics.

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
