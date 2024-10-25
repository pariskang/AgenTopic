# src/data_loader.py

import pandas as pd
import src.config as config

def load_main_data(file_path):
    """
    Load the main dataset from an Excel file.
    Args:
        file_path (str): Path to the main data Excel file.
    Returns:
        tuple: DataFrame and list of documents.
    """
    data = pd.read_excel(file_path)
    
    # Load stopwords
    try:
        with open('stopwords.txt', 'r', encoding='utf-8') as f:
            stopwords = set(line.strip() for line in f)
    except FileNotFoundError:
        print("Warning: stopwords.txt not found. Proceeding without stopwords.")
        stopwords = set()
    
    # Remove @mentions and stopwords while keeping the column name 'Info'
    data['Info'] = data['Info'].apply(lambda x: ' '.join(
        word for word in str(x).split() 
        if not word.startswith('@') and word.lower() not in stopwords
    ))
    
    documents = data['Info'].tolist()
    return data, documents

def load_test_data(file_path):
    """
    Load the test dataset from an Excel file.
    Args:
        file_path (str): Path to the test data Excel file.
    Returns:
        tuple: DataFrame, list of test documents, and list of true labels.
    """
    test_data = pd.read_excel(file_path)
    
    # Load stopwords
    try:
        with open('stopwords.txt', 'r', encoding='utf-8') as f:
            stopwords = set(line.strip() for line in f)
    except FileNotFoundError:
        print("Warning: stopwords.txt not found. Proceeding without stopwords.")
        stopwords = set()
    
    # Remove @mentions and stopwords while keeping the column name 'Info'
    test_data['Info'] = test_data['Info'].apply(lambda x: ' '.join(
        word for word in str(x).split() 
        if not word.startswith('@') and word.lower() not in stopwords
    ))
    
    test_documents = test_data['Info'].tolist()
    test_labels = test_data['Label'].tolist()
    return test_data, test_documents, test_labels
