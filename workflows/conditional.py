from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Literal


class QuadState(TypedDict):
    a:int
    b:int
    c:int

    equation:str
    dicriminant:int
    result:str

def calculate_equation(state:QuadState):
    equation=f"{state['a']}x^2 + {state['b']}x + {state['c']} = 0"
    return {"equation":equation}

def check_discriminant(state:QuadState):
    dicriminant=state["b"]**2-4*state["a"]*state["c"]
    return {"dicriminant":dicriminant}

def realRoots(state:QuadState):
    root1=(-state["b"]+state["dicriminant"]**0.5)/(2*state["a"])
    root2=(-state["b"]-state["dicriminant"]**0.5)/(2*state["a"])
    result=f"The roots are {round(root1,2)} and {round(root2,2)}"
    return {"result":result}

def repeated_root(state:QuadState):
    root=(-state["b"]+state["dicriminant"]**0.5)/(2*state["a"])
    result=f"The root is {round(root,2)}"
    return {"result":result}
def no_real_root(state:QuadState):
    result="No real roots"
    return {"result":result}

def check_condition(state:QuadState)->Literal["realRoots","repeated_root","no_real_root"]:
    if state["dicriminant"]>0:
        return "realRoots"
    elif state["dicriminant"]==0:
        return "repeated_root"
    else:
        return "no_real_root"

graph=StateGraph(QuadState)
graph.add_node("calculate_equation",calculate_equation)
graph.add_node("check_discriminant",check_discriminant)
graph.add_node("realRoots",realRoots)
graph.add_node("repeated_root",repeated_root)
graph.add_node("no_real_root",no_real_root)

graph.add_edge(START,"calculate_equation")
graph.add_edge("calculate_equation","check_discriminant")

graph.add_conditional_edges("check_discriminant",check_condition)
  

graph.add_edge("realRoots",END)
graph.add_edge("repeated_root",END)
graph.add_edge("no_real_root",END)



workflow=graph.compile()
intial_state={"a":4,"b":-5,"c":-4}
ans=workflow.invoke(intial_state)
print(ans)
    

    
    


