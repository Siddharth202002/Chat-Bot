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

# create state
class LLMState(TypedDict,total=False):
    question:str
    answer:str

def llm_node(state:LLMState)->LLMState:
    print(state)
    question=state["question"]
    prompt=f"""
    Answer the following question: {question}
    """
    response=llm.invoke(prompt)
    state["answer"]=response.content
    return state
    

    
    

# create graph
gaph=StateGraph(LLMState)

# add nodes
gaph.add_node("llm_node",llm_node)

# add edges
gaph.add_edge(START,"llm_node")
gaph.add_edge("llm_node",END)

# compile graph
graph=gaph.compile()

# initial state
initial_state={"question":"who is virat kholi"}

# invoke graph
result=graph.invoke(initial_state)
print(result["answer"])
