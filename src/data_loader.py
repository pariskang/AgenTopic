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
    documents = data['Info'].tolist()       # Ensure 'Text' column exists
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
    test_documents = test_data['Info'].tolist()    # Ensure 'Text' column exists
    test_labels = test_data['Label'].tolist()      # Treat 'Label' as single string
    return test_data, test_documents, test_labels
