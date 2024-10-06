# src/utils.py

import re
import numpy as np
from sentence_transformers import SentenceTransformer
from litellm import completion
import src.config as config
import pandas as pd
from bertopic import BERTopic

class CustomEmbeddingModel(SentenceTransformer):
    def __init__(self, model_path):
        """
        Initialize the custom embedding model.

        Args:
            model_path (str): Path to the embedding model.
        """
        super().__init__(model_path)
    
    def embed_documents(self, documents, verbose=False):
        """
        Generate embeddings for a list of documents.

        Args:
            documents (list): List of document texts.
            verbose (bool): Whether to show a progress bar.

        Returns:
            ndarray: Numpy array of embeddings.
        """
        return self.encode(
            documents,
            batch_size=16,
            show_progress_bar=verbose,
            convert_to_numpy=True
        )
    
    def encode_embeddings(self, documents, batch_size=16, show_progress_bar=False, convert_to_numpy=True):
        """
        Alias for embed_documents to maintain compatibility.

        Args:
            documents (list): List of document texts.
            batch_size (int): Batch size for encoding.
            show_progress_bar (bool): Whether to show a progress bar.
            convert_to_numpy (bool): Whether to convert embeddings to numpy arrays.

        Returns:
            ndarray: Numpy array of embeddings.
        """
        return self.embed_documents(documents, verbose=show_progress_bar)

def generate_topic_summaries(topic_model):
    """
    Generate summaries for each topic using GPT-4.

    Args:
        topic_model (BERTopic): Trained BERTopic model.

    Returns:
        dict: Dictionary mapping topic IDs to their summaries.
    """
    topic_summaries = {}
    for topic_id in topic_model.get_topic_info()['Topic']:
        if topic_id == -1:
            continue  # Skip outlier topic
        topic_words = topic_model.get_topic(topic_id)
        if topic_words:
            words = [word for word, _ in topic_words]
            prompt = (
                f"Please provide a brief summary or label for a topic that includes the following keywords: "
                f"{', '.join(words)}."
            )
            # Call GPT-4 to get topic summary
            response = completion(
                model=config.GPT4_MINI_MODEL_NAME,
                messages=[{"content": prompt, "role": "user"}]
            )
            summary = response['choices'][0]['message']['content'].strip()
            topic_summaries[topic_id] = summary
            print(f"Topic {topic_id}: {summary}")
    return topic_summaries

def get_gpt4_feedback(topic_summaries):
    """
    Get GPT-4 feedback on whether to merge or split topics.

    Args:
        topic_summaries (dict): Dictionary of topic summaries.

    Returns:
        str: GPT-4's advice.
    """
    all_topics_summary = "Here are the topics and their summaries:\n"
    for topic_id, summary in topic_summaries.items():
        all_topics_summary += f"Topic {topic_id}: {summary}\n"

    prompt = all_topics_summary + """
Based on the above summaries, please advise if any topics should be merged or split. If so, please list the topics to be merged or split, and explain briefly why.

Please format your response as:

- Topics to Merge: [(Topic IDs to merge), ...]
- Topics to Split: [Topic IDs to split]

Example:

- Topics to Merge: [(1, 2), (3, 4)]
- Topics to Split: [5, 6]
"""
    response = completion(
        model=config.GPT4_MODEL_NAME,
        messages=[{"content": prompt, "role": "user"}]
    )
    advice = response['choices'][0]['message']['content'].strip()
    print("\nGPT-4's advice:")
    print(advice)
    return advice

def parse_gpt4_advice(advice):
    """
    Parse GPT-4's advice to extract topics to merge and split.

    Args:
        advice (str): GPT-4's response.

    Returns:
        tuple: List of topics to merge and list of topics to split.
    """
    topics_to_merge = []
    topics_to_split = []

    # Parse merge suggestions using regex
    merge_match = re.search(r'Topics to Merge:\s*(\[\(.*?\)\])', advice, re.DOTALL)
    if merge_match:
        merge_str = merge_match.group(1)
        try:
            topics_to_merge = eval(merge_str)
        except Exception as e:
            print("Error parsing Topics to Merge:", e)

    # Parse split suggestions using regex
    split_match = re.search(r'Topics to Split:\s*(\[\d+(?:, \d+)*\])', advice, re.DOTALL)
    if split_match:
        split_str = split_match.group(1)
        try:
            topics_to_split = eval(split_str)
        except Exception as e:
            print("Error parsing Topics to Split:", e)
    return topics_to_merge, topics_to_split

def update_topics_after_merging(topic_model, documents, topics_to_merge):
    """
    Merge topics as per GPT-4's suggestions, ensuring topic IDs exist.

    Args:
        topic_model (BERTopic): Trained BERTopic model.
        documents (list): List of document texts.
        topics_to_merge (list): List of tuples indicating topics to merge.

    Returns:
        tuple: Updated BERTopic model and list of topic assignments.
    """
    current_topic_ids = topic_model.get_topic_info()['Topic'].tolist()
    for merge_pair in topics_to_merge:
        if isinstance(merge_pair, (tuple, list)) and len(merge_pair) >= 2:
            # Check if all topic IDs in the pair exist
            if all(topic_id in current_topic_ids for topic_id in merge_pair):
                try:
                    topic_model.merge_topics(documents, list(merge_pair))
                    print(f"Merged topics {merge_pair}")
                except Exception as e:
                    print(f"Error merging topics {merge_pair}: {e}")
            else:
                missing_topics = [tid for tid in merge_pair if tid not in current_topic_ids]
                print(f"Cannot merge topics {merge_pair}: missing topics {missing_topics}")
        else:
            print(f"Invalid merge pair: {merge_pair}")
    # Reassign topics after merging
    topics, _ = topic_model.transform(documents)
    return topic_model, topics

def update_topics_after_splitting(topic_model, documents, topics, topics_to_split):
    """
    Split topics as per GPT-4's suggestions, ensuring sufficient documents.

    Args:
        topic_model (BERTopic): Trained BERTopic model.
        documents (list): List of document texts.
        topics (list): Current topic assignments.
        topics_to_split (list): List of topic IDs to split.

    Returns:
        list: Updated list of topic assignments.
    """
    for topic_id in topics_to_split:
        # Get indices of documents belonging to the topic
        indices = [i for i, t in enumerate(topics) if t == topic_id]
        if not indices:
            print(f"No documents found for topic {topic_id}. Skipping split.")
            continue
        # Extract documents to split
        docs_to_split = [documents[i] for i in indices]
        num_docs = len(docs_to_split)
        if num_docs < 2:
            print(f"Not enough documents to split topic {topic_id} (only {num_docs} documents). Skipping split.")
            continue
        # Perform new topic modeling on these documents
        try:
            sub_topic_model = BERTopic()
            sub_topics, _ = sub_topic_model.fit_transform(docs_to_split)
            # Assign new topic IDs
            new_topic_ids = [f"{topic_id}_{sub_topic}" if sub_topic != -1 else -1 for sub_topic in sub_topics]
            # Update the original topics
            for idx, doc_idx in enumerate(indices):
                topics[doc_idx] = new_topic_ids[idx]
            print(f"Split topic {topic_id} into subtopics")
        except ValueError as ve:
            print(f"ValueError while splitting topic {topic_id}: {ve}")
        except Exception as e:
            print(f"Unexpected error while splitting topic {topic_id}: {e}")
    return topics

def create_custom_embedding_model(model_path):
    """
    Create an instance of the custom embedding model.

    Args:
        model_path (str): Path to the embedding model.

    Returns:
        CustomEmbeddingModel: Instance of the custom embedding model.
    """
    model = CustomEmbeddingModel(model_path)
    return model

def save_final_results(optimal_model_data, data, output_file):
    """
    Assign final topics and summaries to the main dataset and save to an Excel file.

    Args:
        optimal_model_data (dict): Data of the optimal model.
        data (DataFrame): Main dataset.
        output_file (str): Path to the output Excel file.
    """
    # Extract the optimal model
    optimal_topic_model = optimal_model_data['topic_model']

    # Apply the optimal model to the entire main dataset to get topic assignments
    print("\nAssigning topics to the entire main dataset using the optimal model...")
    final_topics, final_probabilities = optimal_topic_model.transform(data['Text'].tolist())

    # Generate Top5 words for each assigned topic
    final_topic_words = []
    for t in final_topics:
        if t == -1:
            final_topic_words.append([])  # Handle outlier topic
        else:
            topic_words = optimal_topic_model.get_topic(t)
            if topic_words:
                top5_words = [word for word, _ in topic_words[:5]]
                final_topic_words.append(top5_words)
            else:
                final_topic_words.append([])

    # Generate summaries for each topic if not already done
    if 'Topic_Summary' not in data.columns or data['Topic_Summary'].isnull().all():
        print("Generating topic summaries...")
        topic_summaries = generate_topic_summaries(optimal_topic_model)
    else:
        topic_summaries = optimal_model_data.get('topic_summaries', {})

    # Assign topics and summaries to the main dataset
    data['Topic'] = final_topics
    data['Topic_Words'] = final_topic_words
    data['Topic_Summary'] = [topic_summaries.get(topic, '') for topic in final_topics]

    # Save initial topic assignments for inspection if this is the first iteration
    if config.NUM_ITERATIONS == 0:
        initial_results = pd.DataFrame({
            'Text': data['Text'],
            'Topic': final_topics,
            'Topic_Words': final_topic_words,
            'Topic_Summary': data['Topic_Summary']
        })
        initial_results.to_excel(config.INITIAL_RESULTS_FILE, index=False)
        print(f"\nInitial topic modeling results have been saved to '{config.INITIAL_RESULTS_FILE}'.")

    # Save to Excel
    data.to_excel(output_file, index=False)
    print(f"\nFinal results have been saved to '{output_file}'.")

        
