# src/topic_modeling.py

from bertopic import BERTopic

def initial_topic_modeling(documents, embedding_model):
    """
    Perform initial topic modeling using BERTopic.

    Args:
        documents (list): List of document texts.
        embedding_model (SentenceTransformer): Embedding model instance.

    Returns:
        tuple: Trained BERTopic model, list of topic assignments, and probabilities.
    """
    topic_model = BERTopic(
        embedding_model=embedding_model,
        language="english",
        calculate_probabilities=True
    )
    topics, probabilities = topic_model.fit_transform(documents)
    return topic_model, topics, probabilities

