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
    title:str
    outline:str
    content:str
    rating:int
    
def create_outline(state:LLMState)->LLMState:
    
    title=state["title"]
    prompt=f"""
    Generate a outline on the following topic: {title}
    """
    response=llm.invoke(prompt)
    state["outline"]=response.content
    return state

def create_content(state:LLMState)->LLMState:
    
    outline=state["outline"]
    prompt=f"""
    Generate a content on the following outline: {outline}
    """
    response=llm.invoke(prompt)
    state["content"]=response.content
    return state

def rate_blog(state:LLMState)->LLMState:
    
    content=state["content"]
    outline=state["outline"]
    prompt=f"""
    Rate the following content {content} on the following outline: {outline}
    make sure rating will in between 1 to 10
    """
    response=llm.invoke(prompt)
    state["rating"]=response.content
    return state



    

    
    

# create graph
gaph=StateGraph(LLMState)

# add nodes
gaph.add_node("create_outline",create_outline)
gaph.add_node("create_content",create_content)
gaph.add_node("rate_blog",rate_blog)

# add edges
gaph.add_edge(START,"create_outline")
gaph.add_edge("create_outline","create_content")
gaph.add_edge("create_content","rate_blog")
gaph.add_edge("rate_blog",END)

# compile graph
graph=gaph.compile()

# initial state
initial_state={"title":"write a blog on virat kholi"}

# invoke graph
result=graph.invoke(initial_state)
print(result["rating"])
