
import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Set environment variables for API keys
os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"
os.environ["OPENAI_API_BASE"] = "OPENAI_API_BASE"

# Data file path
DATA_FILE_PATH = 'data/psoriasis_papers_cleaned.xlsx'
FINAL_DATA_FILE = 'data_with_final_topics.xlsx'

# Embedding model name
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

# Number of iterations for refinement
NUM_ITERATIONS = 3

# Models for GPT-4 interactions
GPT4_MODEL_NAME = 'gpt-4'
GPT4_MINI_MODEL_NAME = 'gpt-4'
CHATGPT_MODEL_NAME = 'gpt-4'
        
