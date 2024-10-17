from dotenv import load_dotenv
import os
from source import retriever

load_dotenv()

if __name__ == "__main__":
    
    # question = "I love trekking on Pune forts in mansoon season"
    question = "multiprocessing in Jupyter"
    documents = retriever.invoke(question)
    
    print({"documents": documents, "question": question})