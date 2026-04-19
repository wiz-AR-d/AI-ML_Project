import os
import re
from typing import TypedDict, List, Annotated, Dict
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
import wikipedia

import base64

# Define the State
class ResearchState(TypedDict):
    query: str
    search_results: str
    extracted_notes: str
    report: str
    error: str
    search_count: int

# Lazy LLM initialization — avoids startup crash when GOOGLE_API_KEY is missing
_llm = None

def get_llm():
    global _llm
    if _llm is None:
        from langchain_google_genai import ChatGoogleGenerativeAI
        import os
        
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set. Please configure it in your deployment environment.")
            
        _llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=api_key)
    return _llm

def search_node(state: ResearchState) -> ResearchState:
    query = state["query"]
    search_count = state.get("search_count", 0)
    
    if search_count > 2:
        return {"error": "Too many searches, aborting.", "search_results": ""}
        
    try:
        search_results = wikipedia.search(query, results=3)
        if not search_results:
            return {"search_results": "No results found from Wikipedia. Try answering based on your internal knowledge.", "search_count": search_count + 1}
        
        formatted_results = []
        for title in search_results:
            try:
                page = wikipedia.page(title, auto_suggest=False)
                formatted_results.append(f"Source: {page.url}\nContent: {page.summary[:1500]}")
            except Exception:
                continue
                
        if not formatted_results:
            return {"error": "Search failed to extract readable content.", "search_results": ""}
            
        final_results = "\n\n".join(formatted_results)
        return {"search_results": final_results, "search_count": search_count + 1, "error": state.get("error", "")}
    except Exception as e:
        return {"error": f"Search failed: {str(e)}", "search_results": ""}

def summarize_node(state: ResearchState) -> ResearchState:
    if state.get("error"):
        return state
        
    results = state["search_results"]
    if not results or results == "No results found.":
        return {"error": "Insufficient search results."}
        
    prompt = f"""
    Analyze the following search results for the query: "{state['query']}".
    Extract key findings, statistics, and facts.
    
    Results:
    {results}
    """
    try:
        response = get_llm().invoke([HumanMessage(content=prompt)])
        return {"extracted_notes": response.content}
    except Exception as e:
        return {"error": f"Summarization failed: {str(e)}"}

def report_node(state: ResearchState) -> ResearchState:
    if state.get("error"):
        return state
        
    prompt = f"""
    Using the extracted findings below, write a comprehensive Markdown research report on the topic: "{state['query']}".
    The report MUST include the following sections exactly:
    # Title
    ## Abstract
    ## Key Findings
    ## Sources (use the URLs from the search results)
    ## Conclusion
    
    Extracted Findings:
    {state['extracted_notes']}
    
    Search Results containing URLs:
    {state['search_results']}
    """
    try:
        response = get_llm().invoke([
            SystemMessage(content="You are an expert autonomous AI research assistant. Provide highly structured markdown reports focusing on accuracy and professional tone."), 
            HumanMessage(content=prompt)
        ])
        return {"report": response.content}
    except Exception as e:
        return {"error": f"Report generation failed: {str(e)}"}

# Build Graph
graph_builder = StateGraph(ResearchState)

graph_builder.add_node("search", search_node)
graph_builder.add_node("summarize", summarize_node)
graph_builder.add_node("report", report_node)

graph_builder.set_entry_point("search")
graph_builder.add_edge("search", "summarize")
graph_builder.add_edge("summarize", "report")
graph_builder.add_edge("report", END)

research_graph = graph_builder.compile()

async def run_agent(query: str) -> dict:
    initial_state = {
        "query": query,
        "search_results": "",
        "extracted_notes": "",
        "report": "",
        "error": "",
        "search_count": 0
    }
    
    final_state = await research_graph.ainvoke(initial_state)
    
    if final_state.get("error"):
        return {"status": "error", "message": final_state["error"]}
        
    return {
        "status": "success",
        "report": final_state.get("report", "No report generated.")
    }
