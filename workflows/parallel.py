from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict



load_dotenv()


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    api_key=os.getenv("gemini-api-key"),
    temperature=0.7
)


class BatsmanState(TypedDict, total=False):
    runs:int
    fours:int
    sixes:int
    strike_rate:float   
    balls:int
    bpb:float
    bp:float
    summary:str


def calculate_sr(state:BatsmanState):
    return {"strike_rate": (state["runs"] / state["balls"]) * 100}


def calculate_bpb(state:BatsmanState):
    boundaries = state["fours"] + state["sixes"]
    if boundaries == 0:
        return {"bpb": 0.0}
    return {"bpb": state["balls"] / boundaries}


def calculate_bp(state:BatsmanState):
    if state["runs"] == 0:
        return {"bp": 0.0}
    boundary_runs = state["fours"] * 4 + state["sixes"] * 6
    return {"bp": (boundary_runs / state["runs"]) * 100}


def summary(state:BatsmanState):
    return {
        "summary": (
            f"Runs: {state['runs']}, Fours: {state['fours']}, Sixes: {state['sixes']}, "
            f"Strike Rate: {state['strike_rate']}, Balls: {state['balls']}, "
            f"BPB: {state['bpb']}, BP: {state['bp']}"
        )
    }


graph = StateGraph(BatsmanState)
graph.add_node('calculate_sr', calculate_sr)
graph.add_node('calculate_bpb', calculate_bpb)
graph.add_node('calculate_bp', calculate_bp)
graph.add_node('summary', summary)

graph.add_edge(START, 'calculate_sr')
graph.add_edge(START, 'calculate_bpb')
graph.add_edge(START, 'calculate_bp')

graph.add_edge('calculate_sr', 'summary')
graph.add_edge('calculate_bpb', 'summary')
graph.add_edge('calculate_bp', 'summary')

graph.add_edge('summary', END)

workflow = graph.compile()

initial_state = {"runs": 100, "balls": 50, "sixes": 4, "fours": 6}
ans = workflow.invoke(initial_state)
print(ans["summary"])




    
    
    
    
    
