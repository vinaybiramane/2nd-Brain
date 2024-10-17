from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI
from graph.state import GraphState
from typing import Any, Dict
from dotenv import load_dotenv

load_dotenv()
llm = AzureChatOpenAI(
        azure_deployment="gpt-4o-mini",  # or your deployment
        api_version="2024-08-01-preview",  # or your api version
        temperature=0)
prompt = hub.pull("rlm/rag-prompt")

generation_chain = prompt | llm | StrOutputParser()


def generate(state: GraphState) -> Dict[str, Any]:
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    generation = generation_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}