# CogniCache: Your AI-Powered Second Brain

CogniCache is an advanced AI-driven system designed to serve as your digital secondary memory. It addresses the challenge of information overload and knowledge retention by creating a personalized, searchable knowledge base from various media sources.

## Project Overview

In today's information-rich world, we often encounter valuable content through articles, videos, and other media. However, retaining and recalling this information can be challenging. CogniCache aims to solve this problem by creating a "second brain" - a system that stores, organizes, and retrieves information when needed, effectively extending your cognitive capabilities.

## Key Features

- **Bookmark Integration**: Extracts and processes URLs from bookmark files.
- **Multi-Source Input**: Handles various media types including web pages, notes, and videos.
- **AI-Powered Processing**: Utilizes GPT-4o Mini for advanced language understanding and generation.
- **Efficient Embedding**: Employs text-embedding-3-large for creating semantic representations of content.
- **Vector Storage**: Uses Chroma DB for efficient storage and retrieval of embedded information.
- **RAG (Retrieval-Augmented Generation) Flow**: Implements LangGraph for sophisticated information retrieval and generation.

## Technical Stack

- **Main Language Model**: GPT-4o Mini
- **Embedding Model**: text-embedding-3-large
- **Vector Database**: ChromaDB
- **Framework**: LangChain and LangGraph
- **Additional Libraries**: BeautifulSoup4 for HTML parsing

## Future Plans

- **Multimodal Capabilities**: Planning to integrate CLIP or GPT-4 Vision model for processing image and video content.

## Getting Started

1. Clone the repository
2. Install required dependencies:
3. Set up your environment variables in a `.env` file
4. Run the flow.py script

## Contributing

Contributions to CogniCache are welcome!

