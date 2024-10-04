from bertopic import BERTopic

def initial_topic_modeling(documents, embedding_model):
    # Initialize BERTopic model with the embedding model
    topic_model = BERTopic(
        embedding_model=embedding_model,
        language="english",
        calculate_probabilities=True
    )
    topics, probabilities = topic_model.fit_transform(documents)
    return topic_model, topics, probabilities
