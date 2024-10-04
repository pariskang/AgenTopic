import re
import numpy as np
from litellm import completion
import src.config as config

def generate_topic_summaries(topic_model):
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
    topics_to_merge = []
    topics_to_split = []

    # Parse the advice using regular expressions
    merge_match = re.search(r'Topics to Merge:\s*(\[.*?\])', advice, re.DOTALL)
    if merge_match:
        merge_str = merge_match.group(1)
        try:
            topics_to_merge = eval(merge_str)
        except Exception as e:
            print("Error parsing Topics to Merge:", e)

    split_match = re.search(r'Topics to Split:\s*(\[.*?\])', advice, re.DOTALL)
    if split_match:
        split_str = split_match.group(1)
        try:
            topics_to_split = eval(split_str)
        except Exception as e:
            print("Error parsing Topics to Split:", e)
    return topics_to_merge, topics_to_split

def update_topics_after_merging(topic_model, documents, topics_to_merge, embedding_model):
    for merge_pair in topics_to_merge:
        if isinstance(merge_pair, (tuple, list)) and len(merge_pair) >= 2:
            merge_topics = list(merge_pair)
            try:
                topic_model.merge_topics(documents, merge_topics)
                print(f"Merged topics {merge_topics}")
            except Exception as e:
                print(f"Error merging topics {merge_topics}: {e}")
        else:
            print(f"Invalid merge pair: {merge_pair}")
    topics, _ = topic_model.transform(documents)
    return topic_model, topics

def update_topics_after_splitting(topic_model, documents, topics, topics_to_split, embedding_model):
    for topic_id in topics_to_split:
        # Get indices of documents belonging to the topic
        indices = [i for i, t in enumerate(topics) if t == topic_id]
        if not indices:
            continue
        # Extract documents to split
        docs_to_split = [documents[i] for i in indices]
        # Perform new topic modeling on these documents
        sub_topic_model = BERTopic(
            embedding_model=embedding_model,
            language="english",
            calculate_probabilities=True
        )
        sub_topics, _ = sub_topic_model.fit_transform(docs_to_split)
        # Update the original topics
        for idx, doc_idx in enumerate(indices):
            topics[doc_idx] = f"{topic_id}_{sub_topics[idx]}"
        print(f"Split topic {topic_id} into subtopics")
    return topics

def create_custom_embedding_model(model_path):
    encoder = AutoModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    class CustomEmbeddingModel:
        def __init__(self, model, tokenizer):
            self.model = model
            self.tokenizer = tokenizer

        def embed_documents(self, documents, verbose=False):
            embeddings = []
            batch_size = 16
            for i in range(0, len(documents), batch_size):
                batch_texts = documents[i:i+batch_size]
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                )
                with torch.no_grad():
                    outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state.mean(dim=1)
                embeddings.extend(batch_embeddings.cpu().numpy())
            return np.array(embeddings)

    custom_embedding_model = CustomEmbeddingModel(model=encoder, tokenizer=tokenizer)
    return custom_embedding_model

def save_final_results(optimal_model_data, data, output_file):
    # Assign the final topics and summaries to the data
    final_topics = optimal_model_data['topics']
    final_topic_summaries = optimal_model_data['topic_summaries']
    data['Topic'] = final_topics
    data['Topic_Summary'] = [final_topic_summaries.get(topic, '') for topic in final_topics]

    # Save the data to an Excel file
    data.to_excel(output_file, index=False)
    print(f"\nFinal results have been saved to '{output_file}'.")

        