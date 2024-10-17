from dotenv import load_dotenv
from bs4 import BeautifulSoup
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PlaywrightURLLoader
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv()
embeddings = AzureOpenAIEmbeddings(model = 'text-embedding-3-large',
                                   azure_endpoint = os.getenv("EMB_AZURE_OPENAI_ENDPOINT"),
                                   api_version = os.getenv('EMB_AZURE_OPENAI_API_VERSION'))
bookmark_file = 'bookmarks_10_16_24.html'
select_urls_range = range(7,100)

def extract_urls_from_bookmarks(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')
    
    urls = [a['href'] for a in soup.find_all('a', href=True)]
    return urls

def url_to_docs(url_list):
    
    loader = PlaywrightURLLoader(urls=url_list)
    docs = loader.load()
    docs = [doc for doc in docs if doc.page_content != 'Page not found\n\nReturn home?' ]
    print("Total urls selected- ", len(url_list), "Page not found for - ", len(url_list) - len(docs))

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=50
    )
    doc_splits = text_splitter.split_documents(docs)

    return doc_splits


url_list = extract_urls_from_bookmarks(bookmark_file)
# Print the first few URLs to verify
print(f"Total URLs extracted: {len(url_list)}")
url_list = [ url for i, url in enumerate(url_list) if i in select_urls_range]

doc_splits = url_to_docs(url_list)

# print([type(x) for x in doc_splits])

vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=embeddings,
    persist_directory="./.chroma",
)

retriever = Chroma(
    collection_name="rag-chroma",
    persist_directory="./.chroma",
    embedding_function=embeddings,
    
).as_retriever(search_type = 'similarity_score_threshold', search_kwargs = {'score_threshold' : 0.7, 'k':3 })
