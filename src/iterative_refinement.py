# src/iterative_refinement.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from bertopic import BERTopic
from litellm import completion
from src.fine_tuning import fine_tune_model
from src.evaluation import evaluate_model, select_optimal_model
from src.utils import (
    generate_topic_summaries,
    get_gpt4_feedback,
    parse_gpt4_advice,
    update_topics_after_merging,
    update_topics_after_splitting,
    create_custom_embedding_model,
)
import src.config as config

def iterative_refinement(data, documents, topic_model, topics, probabilities, embedding_model):
    """
    Performs iterative refinement of the topic model based on GPT-4 feedback.
    """
    models = []
    metrics = []
    topic_summaries_list = []

    for iteration in range(1, config.NUM_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---\n")
        
        # Step 1: Generate Topic Summaries
        topic_summaries = generate_topic_summaries(topic_model)

        # Store the summaries for this iteration
        topic_summaries_list.append(topic_summaries)

        # Step 2: Get GPT-4 Feedback for Merging or Splitting Topics
        advice = get_gpt4_feedback(topic_summaries)

        # Step 3: Parse GPT-4's Feedback and Implement Suggestions
        topics_to_merge, topics_to_split = parse_gpt4_advice(advice)

        # Step 4: Update topics based on GPT-4's advice
        # Set the embedding model back to the current embedding model
        topic_model.embedding_model = embedding_model

        # Update topics based on merging suggestions
        topic_model, topics = update_topics_after_merging(
            topic_model, documents, topics_to_merge
        )

        # Update topics based on splitting suggestions
        topics = update_topics_after_splitting(
            topic_model, documents, topics, topics_to_split
        )

        # Step 5: Fine-Tune the Embedding Model Based on New Topics
        model_name = config.EMBEDDING_MODEL_NAME
        unique_topics = list(set(topics))
        labels = [unique_topics.index(t) for t in topics]
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            documents, labels, test_size=0.1, random_state=42
        )

        fine_tuned_model_path = fine_tune_model(
            iteration, model_name, train_texts, train_labels, test_texts, test_labels, len(unique_topics)
        )

        # Step 6: Recompute Embeddings with Fine-Tuned Model
        custom_embedding_model = create_custom_embedding_model(fine_tuned_model_path)

        # Step 7: Update the embedding model in the topic_model
        topic_model.embedding_model = custom_embedding_model

        # Step 8: Refit Topic Model with New Embeddings
        # Let BERTopic compute the embeddings using the updated embedding model
        try:
            topics, probabilities = topic_model.fit_transform(documents)
        except ValueError as ve:
            print(f"ValueError during fit_transform in iteration {iteration}: {ve}")
            print("Skipping this iteration.")
            continue
        except Exception as e:
            print(f"Unexpected error during fit_transform in iteration {iteration}: {e}")
            print("Skipping this iteration.")
            continue

        # Step 9: Evaluate the Model
        evaluation_metrics = evaluate_model(
            iteration, custom_embedding_model, documents, topics
        )
        metrics.append(evaluation_metrics)

        # Store the model data
        models.append({
            'iteration': iteration,
            'topic_model': topic_model,
            'embeddings': None,  # Not storing embeddings as BERTopic manages them
            'topics': topics,
            'topic_summaries': topic_summaries
        })

        # Update the embedding model for the next iteration
        embedding_model = custom_embedding_model

    # Return all models and metrics for selection
    return {
        'models': models,
        'metrics': metrics,
        'topic_summaries_list': topic_summaries_list
    }
        
