# src/topic_modeling.py

from bertopic import BERTopic

def initial_topic_modeling(documents, embedding_model):
    """
    Performs initial topic modeling using BERTopic.
    """
    topic_model = BERTopic(
        embedding_model=embedding_model,
        language="english",
        calculate_probabilities=True
    )
    topics, probabilities = topic_model.fit_transform(documents)
    return topic_model, topics, probabilities

