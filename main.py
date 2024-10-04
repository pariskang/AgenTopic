###FulPhil

import src.config as config
from src.data_loader import load_data
from src.embedding_model import initialize_embedding_model
from src.topic_modeling import initial_topic_modeling
from src.iterative_refinement import iterative_refinement
from src.evaluation import select_optimal_model
from src.utils import save_final_results

def main():
    # Section 2: Load Data
    data, documents = load_data(config.DATA_FILE_PATH)

    # Section 3: Initialize Embedding Model
    embedding_model = initialize_embedding_model(config.EMBEDDING_MODEL_NAME)

    # Section 4: Initial Topic Modeling with BERTopic
    topic_model, topics, probabilities = initial_topic_modeling(documents, embedding_model)

    # Section 5: Iterative Refinement with GPT-4 Feedback
    final_model_data = iterative_refinement(
        data, documents, topic_model, topics, probabilities, embedding_model
    )

    # Section 6: Select Optimal Model with GPT-4
    optimal_model_data = select_optimal_model(final_model_data)

    # Section 7: Save Final Results
    save_final_results(optimal_model_data, data, config.FINAL_DATA_FILE)

if __name__ == "__main__":
    main()
        