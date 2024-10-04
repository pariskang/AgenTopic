from sentence_transformers import SentenceTransformer

def initialize_embedding_model(model_name):
    # Initialize the embedding model
    embedding_model = SentenceTransformer(model_name)
    return embedding_model
        