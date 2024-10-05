import pandas as pd
import src.config as config
def load_data(file_path):
    # Load data from Excel file
    data = pd.read_excel(file_path)
    documents = data['Info'].astype(str).tolist()
    return data, documents
        
