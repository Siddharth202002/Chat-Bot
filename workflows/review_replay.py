from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict,Literal
from pydantic import Field,BaseModel




load_dotenv()



llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    api_key=os.getenv("gemini-api-key"),
    temperature=0.7
)


class Scentement_Schema(BaseModel):
    scentement:Literal["positive","negative"]=Field(description="The scentement of the review")

class DiagnosisSchema(BaseModel):
    issue_type: Literal["UX", "Performance", "Bug", "Support", "Other"] = Field(description="The category of the issue")
    tone: Literal["angry", "frustrated", "disappointed", "calm"] = Field(description="The emotional tone of the review")
    urgency: Literal["low", "medium", "high"] = Field(description="How urgent or critical the issue is")

class Review_state(TypedDict):
    review:str
    scentement:Literal["positive","negative"]
    dignose_key:DiagnosisSchema
    response:str
    
    
llm_with_schema=llm.with_structured_output(Scentement_Schema)
llm_with_diagnosis=llm.with_structured_output(DiagnosisSchema)

def find_scentement(state:Review_state):
    prompt=f"""find the scentement of the review {state['review']}"""
    response=llm_with_schema.invoke(prompt)
    return {"scentement":response.scentement}

def check_condition(state:Review_state)->Literal["positive_response","run_diagnosis"]:
    if state["scentement"]=="positive":
        return "positive_response"
    else:
        return "run_diagnosis"

def positive_response(state:Review_state):
    prompt = f"""Write a warm thank-you message in response to this review:

"{state['review']}"

Also, kindly ask the user to leave feedback on our website."""
    response = llm.invoke(prompt).content
    return {'response': response}

def run_diagnosis(state:Review_state):
    prompt = f"""Diagnose this negative review:

{state['review']}

Return issue_type, tone, and urgency."""
    response = llm_with_diagnosis.invoke(prompt)
    return {'dignose_key': response}

def negative_response(state:Review_state):
    diagnosis = state["dignose_key"]
    prompt=f"""
    Generate a polite and helpful response to this negative review: {state['review']}
    The issue type is {diagnosis.issue_type}, the tone is {diagnosis.tone}, and the urgency is {diagnosis.urgency}.
    """
    response=llm.invoke(prompt)
    return {"response":response.content}



graph=StateGraph(Review_state)

graph.add_node("find_scentement",find_scentement)
graph.add_node("positive_response",positive_response)
graph.add_node("run_diagnosis",run_diagnosis)
graph.add_node("negative_response",negative_response)

graph.add_edge(START,"find_scentement")
graph.add_conditional_edges("find_scentement",check_condition)

graph.add_edge("positive_response",END)
graph.add_edge("run_diagnosis","negative_response")
graph.add_edge("negative_response",END)


workflow=graph.compile()

initial_state={"review":"The app is very slow and keeps crashing. I am very frustrated."}

result=workflow.invoke(initial_state)
print(result)
