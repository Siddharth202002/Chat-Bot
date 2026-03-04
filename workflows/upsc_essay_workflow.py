from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict,Annotated
from pydantic import Field,BaseModel
import operator



load_dotenv()

class Evavluation_Schema(BaseModel):
    feedback:str=Field(description="detailed feedback for the essay")
    score:int=Field(description="give score out of 10",ge=0,le=10)

class UPSCState(TypedDict):
    essay: str
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    overall_feedback: str
    individual_scores: Annotated[list[int],operator.add]
    average_score:float



llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    api_key=os.getenv("gemini-api-key"),
    temperature=0.7
)
llm_with_schema=llm.with_structured_output(Evavluation_Schema)


def language_feedback(state:UPSCState):
    
    prompt=f"""evaluate the language quality of the essay and give feedback and score out of 10 
    essay: {state["essay"]}"""
    response=llm_with_schema.invoke(prompt)
    return {"language_feedback": response.feedback, "individual_scores": [response.score]}

def analysis_feedback(state:UPSCState):
    prompt=f"""evaluate the analysis quality of the essay and give feedback and score out of 10 
    essay: {state["essay"]}"""
    response=llm_with_schema.invoke(prompt)
    return {"analysis_feedback": response.feedback, "individual_scores": [response.score]}
    
def clarity_feedback(state:UPSCState):
    prompt=f"""evaluate the clarity quality of the essay and give feedback and score out of 10 
    essay: {state["essay"]}"""
    response=llm_with_schema.invoke(prompt)
    return {"clarity_feedback": response.feedback, "individual_scores": [response.score]}
    
def overall_feedback(state:UPSCState):
    prompt=f"""evaluate the overall quality of the essay on the basics of the langauge_feedback {state["language_feedback"]},analysis_feedback {state["analysis_feedback"]},clarity_feedback {state["clarity_feedback"]}
    essay: {state["essay"]}"""

    response=llm.invoke(prompt)
    

    average_score = (
        sum(state["individual_scores"]) / len(state["individual_scores"])
        if state["individual_scores"]
        else 0.0
    )

    return {"overall_feedback": response.content, "average_score": average_score}

    
    



graph=StateGraph(UPSCState)
graph.add_node("language_feedback",language_feedback)
graph.add_node("analysis_feedback",analysis_feedback)
graph.add_node("clarity_feedback",clarity_feedback)
graph.add_node("overall_feedback",overall_feedback)


# edges 

graph.add_edge(START,"language_feedback")
graph.add_edge(START,"analysis_feedback")
graph.add_edge(START,"clarity_feedback")
graph.add_edge("language_feedback","overall_feedback")
graph.add_edge("analysis_feedback","overall_feedback")
graph.add_edge("clarity_feedback","overall_feedback")
graph.add_edge("overall_feedback",END)



initial_state={"essay":"Virat Kohli is one of the greatest cricketers in the world and a proud representative of Indian cricket. He was born on 5 November 1988 in Delhi, India, and developed an interest in cricket at a very young age. His talent became clear when he played in youth tournaments and later captained India to victory in the 2008 Under-19 World Cup. After this success, he was selected for the Indian national team and soon became a key player. Kohli is known for his aggressive batting style, strong technique, and ability to chase targets under pressure. He has scored many centuries in all formats of the game and has broken several records. Apart from batting, he is also an excellent fielder and a motivating leader. He served as the captain of the Indian team in all formats and led with passion and confidence. Kohli is admired for his fitness, discipline, and dedication to the sport. He has inspired many young cricketers in India and around the world. Off the field, he is known for his charity work and sportsmanship. His journey from a young boy in Delhi to an international cricket legend is truly inspiring. Virat Kohli remains a symbol of hard work, determination, and excellence in modern cricket."}


workflow=graph.compile()

result=workflow.invoke(initial_state)
print(result)




