import uuid
import time
import asyncio
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Body, HTTPException, BackgroundTasks
import google.generativeai as genai
from google.generativeai import GenerativeModel
from pydantic import BaseModel, Field, validator

from config import GOOGLE_API_KEY, SERVER_HOST, SERVER_PORT
from models.data_models import (
    ResearchPaper, AbstractExplanation, EnhancedAbstractExplanation,
    ResearchQuery, Task
)
from agents.researcher import ResearcherAgent
from agents.writer import WriterAgent
from agents.quality_assurance import QualityAssuranceAgent, EnhancedQualityAssuranceAgent
from agents.chat_qa import ChatQAAgent
from tools.vector_db import search_papers

# Configure Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize FastAPI app
app = FastAPI(title="Multi-Agent Coordination Protocol Server")

@app.post("/task/researcher/search", response_model=Dict[str, Any])
async def create_research_task(query: ResearchQuery = Body(...), background_tasks: BackgroundTasks = None):
    """Create a research task to search for papers"""
    task_id = str(uuid.uuid4())
    task = Task(
        id=task_id,
        agent="researcher",
        action="search_papers",
        input_data={"query": query.query, "num_papers": query.num_papers}
    )
    
    # This would typically be stored in a database
    # For demo purposes, we'll execute it immediately
    gemini_model = GenerativeModel('gemini-1.5-pro')
    researcher = ResearcherAgent(gemini_model)
    
    try:
        papers = await researcher.search_papers(query.query, query.num_papers)
        return {
            "task_id": task_id,
            "status": "completed",
            "result": [paper.model_dump() for paper in papers]
        }
    except Exception as e:
        return {
            "task_id": task_id,
            "status": "failed",
            "error": str(e)
        }

@app.post("/task/writer/explain", response_model=Dict[str, Any])
async def create_writer_task(paper: ResearchPaper = Body(...)):
    """Create a task for the writer to explain an abstract with enhanced QA"""
    task_id = str(uuid.uuid4())
    task = Task(
        id=task_id,
        agent="writer",
        action="explain_abstract",
        input_data={"paper": paper.model_dump()}
    )
    
    gemini_model = GenerativeModel('gemini-1.5-pro')
    writer = WriterAgent(gemini_model)
    
    try:
        # Generate initial explanation
        explanation = await writer.explain_abstract(paper)
        
        # Use enhanced QA agent for verification
        qa_agent = EnhancedQualityAssuranceAgent(gemini_model)
        verified_explanation = await qa_agent.evaluate_explanation(paper, explanation)
        
        # Create enhanced explanation with metadata
        enhanced_explanation = EnhancedAbstractExplanation.from_basic(
            verified_explanation, 
            metadata=getattr(verified_explanation, 'metadata', {})
        )
        
        return {
            "task_id": task_id,
            "status": "completed",
            "result": enhanced_explanation.model_dump(),
            "quality_score": enhanced_explanation.quality_score,
            "evaluation_summary": enhanced_explanation.get_evaluation_summary()
        }
    except Exception as e:
        return {
            "task_id": task_id,
            "status": "failed",
            "error": str(e)
        }

@app.post("/batch/process", response_model=Dict[str, Any])
async def batch_process_papers(papers: List[ResearchPaper] = Body(...)):
    """Process multiple papers with writing and QA in parallel"""
    batch_id = str(uuid.uuid4())
    
    gemini_model = GenerativeModel('gemini-1.5-pro')
    writer = WriterAgent(gemini_model)
    qa_agent = EnhancedQualityAssuranceAgent(gemini_model)
    
    results = []
    tasks = []
    
    # Create tasks for initial explanations
    for paper in papers:
        task = asyncio.create_task(writer.explain_abstract(paper))
        tasks.append((paper, task))
    
    # Process explanations
    paper_explanations = []
    for paper, task in tasks:
        try:
            explanation = await task
            paper_explanations.append((paper, explanation))
        except Exception as e:
            print(f"Failed to generate explanation for {paper.title}: {str(e)}")
            # Create a placeholder explanation
            error_explanation = AbstractExplanation(
                paper_id=paper.id,
                explanation=f"Error generating explanation: {str(e)}",
                quality_score=0.0
            )
            results.append({
                "paper": paper.model_dump(),
                "explanation": error_explanation.model_dump(),
                "status": "error",
                "error": str(e)
            })
    
    # Process QA in batches to manage rate limits
    batch_size = 3  # Process in small batches
    for i in range(0, len(paper_explanations), batch_size):
        batch = paper_explanations[i:i+batch_size]
        verified_batch = await qa_agent.bulk_evaluation(batch)
        
        for j, verified_explanation in enumerate(verified_batch):
            paper = batch[j][0]
            results.append({
                "paper": paper.model_dump(),
                "explanation": verified_explanation.model_dump(),
                "status": "completed",
                "quality_score": verified_explanation.quality_score
            })
    
    return {
        "batch_id": batch_id,
        "total_papers": len(papers),
        "completed": len(results),
        "results": results
    }

@app.get("/qa/analytics", response_model=Dict[str, Any])
async def get_qa_analytics(paper_id: Optional[str] = None):
    """Get analytics about QA performance"""
    gemini_model = GenerativeModel('gemini-1.5-pro')
    qa_agent = EnhancedQualityAssuranceAgent(gemini_model)
    
    insights = qa_agent.get_improvement_insights(paper_id)
    return {
        "timestamp": time.time(),
        "insights": insights
    }

@app.get("/search", response_model=List[Dict[str, Any]])
async def search_papers_in_db(query: str, limit: int = 5):
    """Search for papers in the ChromaDB using RAG"""
    return search_papers(query, limit)

# Chat models
class ChatQuestion(BaseModel):
    question: str
    session_id: Optional[str] = None
    max_context_papers: int = 3
    
    @validator('max_context_papers')
    def validate_max_papers(cls, v):
        if v < 1:
            return 1
        if v > 10:
            return 10
        return v

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    session_id: str
    context_papers_count: int = 0
    processing_time: float = 0.0

# Chat endpoints
@app.post("/chat/question", response_model=ChatResponse)
async def answer_question(chat_question: ChatQuestion = Body(...)):
    """Answer a research question using the RAG approach with the paper database"""
    gemini_model = GenerativeModel('gemini-1.5-pro')
    chat_agent = ChatQAAgent(gemini_model)
    
    result = await chat_agent.answer_question(
        question=chat_question.question,
        session_id=chat_question.session_id,
        max_context_papers=chat_question.max_context_papers
    )
    
    return result

@app.delete("/chat/session/{session_id}", response_model=Dict[str, Any])
async def clear_chat_session(session_id: str):
    """Clear a specific chat session history"""
    gemini_model = GenerativeModel('gemini-1.5-pro')
    chat_agent = ChatQAAgent(gemini_model)
    
    success = chat_agent.clear_conversation(session_id)
    return {
        "success": success,
        "session_id": session_id,
        "message": "Conversation history cleared" if success else "Session not found"
    }

def start_server():
    """Start the server using uvicorn"""
    import uvicorn
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
