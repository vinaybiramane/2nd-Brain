from dotenv import load_dotenv

from langgraph.graph import END, StateGraph
from graph.nodes import generate, relevance, retrieve, web_search
from graph.state import GraphState
from guard_rails import answer_grader, hallucination_grader


load_dotenv()

RETRIEVE = "retrieve"
RELEVANCE = "relevance"
GENERATE = "generate"
WEBSEARCH = "websearch"

def decide_to_generate(state):
    print("---ASSESS GRADED DOCUMENTS---")

    if len(state["documents"]) == 0 :
        print(
            "---DECISION: NO DOCUMENT IS RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return WEBSEARCH
    else:
        print("---DECISION: GENERATE---")
        return GENERATE


def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        input = {"documents": documents, "generation": generation}
    )

    if score.binary_score:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke(input = {"question": question, "generation": generation})
        if score.binary_score:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"


workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve.retrieve)
workflow.add_node(RELEVANCE, relevance.relevance)
workflow.add_node(GENERATE, generate.generate)
workflow.add_node(WEBSEARCH, web_search.web_search)

workflow.set_entry_point(RETRIEVE)
workflow.add_edge(RETRIEVE, RELEVANCE)
workflow.add_conditional_edges(
    RELEVANCE,
    decide_to_generate,
    {
        WEBSEARCH: WEBSEARCH,
        GENERATE: GENERATE,
    },
)
workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        "not supported": GENERATE,
        "useful": END,
        "not useful": WEBSEARCH,
    },
)
workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="graph.png")

response = app.invoke(input={"question":"Tell me about Pune forts"})

print(response)
print("\n-----------------Generation-----------------\n")
print(response['generation'])