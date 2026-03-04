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
class BMI(TypedDict,total=False):
    weight:int
    height:float
    bmi:float
    label:str


def calculate_bmi(state:BMI)->BMI:
    bmi=state["weight"]/(state["height"]*state["height"])
    state["bmi"]=round(bmi,2)
    return state

def label_bmi(state:BMI)->BMI:
    if state["bmi"]<18.5:
        state["label"]="underweight"
    elif state["bmi"]<25:
        state["label"]="normal"
    elif state["bmi"]<30:
        state["label"]="overweight"
    else:
        state["label"]="obesity"
    return state
    
    

# create graph
gaph=StateGraph(BMI)

# add nodes
gaph.add_node("calculate_bmi",calculate_bmi)
gaph.add_node("label_bmi",label_bmi)

# add edges
gaph.add_edge(START,"calculate_bmi")
gaph.add_edge("calculate_bmi","label_bmi")
gaph.add_edge("label_bmi",END)

# compile graph
graph=gaph.compile()

# initial state
initial_state={"weight":90,"height":1.75}

# invoke graph
result=graph.invoke(initial_state)
print(result)
