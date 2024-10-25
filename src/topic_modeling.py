# src/topic_modeling.py

from bertopic import BERTopic
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

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

"""
def lda_topic_modeling(documents, n_topics=10, max_features=1000, max_iter=10):
    # Perform topic modeling using Latent Dirichlet Allocation (LDA).
    Args:
        documents (list): List of document texts.
        n_topics (int): Number of topics to extract.
        max_features (int): Maximum number of features for vocabulary.
        max_iter (int): Maximum number of iterations.
    Returns:
        tuple: Trained LDA model, document-topic matrix, feature names.
    
    # Create document-term matrix
    vectorizer = CountVectorizer(max_features=max_features)
    doc_term_matrix = vectorizer.fit_transform(documents)
    
    # Initialize and train LDA model
    lda_model = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=max_iter,
        learning_method='online'
    )
    
    # Fit the model and transform documents
    doc_topics = lda_model.fit_transform(doc_term_matrix)
    
    return lda_model, doc_topics, vectorizer.get_feature_names_out()

def nmf_topic_modeling(documents, n_topics=158, max_features=1000):
    # Perform topic modeling using Non-negative Matrix Factorization (NMF).
    Args:
        documents (list): List of document texts.
        n_topics (int): Number of topics to extract.
        max_features (int): Maximum number of features for vocabulary.
    Returns:
        tuple: Trained NMF model, document-topic matrix, feature names.
    
    # Create TF-IDF matrix
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    
    # Initialize and train NMF model
    nmf_model = NMF(
        n_components=n_topics,
        init='nndsvd'  # Use NNDSVD initialization for better stability
    )
    
    # Fit the model and transform documents
    doc_topics = nmf_model.fit_transform(tfidf_matrix)
    
    return nmf_model, doc_topics, tfidf_vectorizer.get_feature_names_out()

def get_top_terms_per_topic(model, feature_names, n_top_words=158):
    # Get top terms for each topic from LDA or NMF model.
    Args:
        model: Trained LDA or NMF model.
        feature_names (list): List of feature names.
        n_top_words (int): Number of top words to return per topic.
    Returns:
        list: List of top terms for each topic.
    
    top_terms = []
    for topic_idx, topic in enumerate(model.components_):
        top_terms_idx = topic.argsort()[:-n_top_words-1:-1]
        top_terms.append([feature_names[i] for i in top_terms_idx])
    return top_terms
"""
