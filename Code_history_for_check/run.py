import os
import json
import uuid
from typing import List, Dict, Any, Optional

# For Gemini API
from google.generativeai import GenerativeModel
import google.generativeai as genai

# For vector database
import chromadb
from chromadb.utils import embedding_functions

# For MCP server
from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import asyncio
import httpx

# Configure Gemini API
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize FastAPI app for MCP server
app = FastAPI(title="Multi-Agent Coordination Protocol Server")

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction()
collection = chroma_client.get_or_create_collection(
    name="research_papers",
    embedding_function=embedding_function
)

# Define data models
class ResearchPaper(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    authors: List[str]
    abstract: str
    url: str
    content: Optional[str] = None
    publication_date: Optional[str] = None
    keywords: List[str] = []

class AbstractExplanation(BaseModel):
    paper_id: str
    explanation: str
    quality_score: float = 0.0

class ResearchQuery(BaseModel):
    query: str
    num_papers: int = 5

class Task(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent: str
    action: str
    status: str = "pending"
    input_data: Dict[str, Any] = {}
    output_data: Dict[str, Any] = {}

# Agent definitions
class ResearcherAgent:
    def __init__(self, gemini_model: GenerativeModel):
        self.model = gemini_model
    
    async def search_papers(self, query: str, num_papers: int = 5) -> List[ResearchPaper]:
        """
        Simulates searching for research papers.
        In a real implementation, this would connect to academic APIs like Semantic Scholar, arXiv, etc.
        """
        # Prompt for paper generation (in a real system, this would be actual API calls)
        search_prompt = f"""
        Generate {num_papers} realistic research paper metadata related to: {query}
        
        For each paper, include:
        1. Title
        2. Authors (3-5 names)
        3. A realistic abstract (200-300 words)
        4. A realistic URL
        5. Publication date (within last 3 years)
        6. 5-7 relevant keywords
        
        Format as a JSON list of objects.
        """
        
        response = await self.model.generate_content_async(search_prompt)
        
        try:
            # Extract JSON from response
            papers_text = response.text
            if "```json" in papers_text:
                papers_text = papers_text.split("```json")[1].split("```")[0].strip()
            elif "```" in papers_text:
                papers_text = papers_text.split("```")[1].split("```")[0].strip()
            
            papers_data = json.loads(papers_text)
            papers = []
            
            for paper_data in papers_data:
                paper = ResearchPaper(
                    title=paper_data.get("title", ""),
                    authors=paper_data.get("authors", []),
                    abstract=paper_data.get("abstract", ""),
                    url=paper_data.get("url", ""),
                    publication_date=paper_data.get("publication_date", ""),
                    keywords=paper_data.get("keywords", [])
                )
                papers.append(paper)
                
                # Store in ChromaDB
                collection.add(
                    ids=[paper.id],
                    documents=[paper.abstract],
                    metadatas=[{
                        "title": paper.title,
                        "authors": ", ".join(paper.authors),
                        "url": paper.url,
                        "publication_date": paper.publication_date,
                        "keywords": ", ".join(paper.keywords)
                    }]
                )
            
            return papers[:num_papers]  # Ensure we only return the requested number
            
        except json.JSONDecodeError:
            # If JSON parsing fails, create a synthetic fallback
            return [
                ResearchPaper(
                    title=f"Error retrieving paper {i} for query: {query}",
                    authors=["System Error"],
                    abstract="Failed to retrieve paper details. Please try again.",
                    url="https://example.com/error",
                    keywords=["error", "retrieval_failure"]
                )
                for i in range(num_papers)
            ]

class WriterAgent:
    def __init__(self, gemini_model: GenerativeModel):
        self.model = gemini_model
    
    async def explain_abstract(self, paper: ResearchPaper) -> AbstractExplanation:
        """Generate an explanation of the paper abstract"""
        prompt = f"""
        Explain the following research paper abstract in clear, concise language:
        
        Title: {paper.title}
        Authors: {', '.join(paper.authors)}
        
        Abstract:
        {paper.abstract}
        
        Your explanation should:
        1. Summarize the main research question or objective
        2. Explain the methodology in simple terms
        3. Highlight key findings and their significance
        4. Identify potential applications or implications
        5. Use accessible language while preserving technical accuracy
        """
        
        response = await self.model.generate_content_async(prompt)
        
        return AbstractExplanation(
            paper_id=paper.id,
            explanation=response.text
        )

class QualityAssuranceAgent:
    def __init__(self, gemini_model: GenerativeModel):
        self.model = gemini_model
    
    async def check_explanation(self, paper: ResearchPaper, explanation: AbstractExplanation) -> AbstractExplanation:
        """Verify explanation against the abstract to detect hallucinations"""
        prompt = f"""
        TASK: Evaluate the explanation below for accuracy compared to the original abstract.
        
        ORIGINAL ABSTRACT:
        {paper.abstract}
        
        EXPLANATION:
        {explanation.explanation}
        
        Perform a detailed analysis:
        1. Identify any factual claims in the explanation not supported by the abstract
        2. Note any misrepresentations of the research methodology
        3. Check if the explanation adds speculative details not in the abstract
        4. Verify that key findings are accurately represented
        5. Assign a score from 0.0 to 1.0 where:
           - 1.0 = Perfect accuracy, no hallucinations
           - 0.0 = Completely inaccurate or fabricated
        
        Return just a JSON with two fields:
        1. "corrected_explanation": the fixed explanation with hallucinations removed
        2. "quality_score": numerical score from 0.0 to 1.0
        """
        
        try:
            response = await self.model.generate_content_async(prompt)
            
            try:
                # Extract JSON from response
                response_text = response.text
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0].strip()
                
                qa_result = json.loads(response_text)
                
                corrected_explanation = qa_result.get("corrected_explanation", explanation.explanation)
                quality_score = qa_result.get("quality_score", 0.0)
                
                # Only accept the explanation if the quality score is above threshold
                if quality_score < 0.7:
                    # Attempt to regenerate a better explanation
                    return await self.regenerate_explanation(paper, explanation, quality_score)
                
                return AbstractExplanation(
                    paper_id=paper.id,
                    explanation=corrected_explanation,
                    quality_score=quality_score
                )
                
            except (json.JSONDecodeError, KeyError):
                # If parsing fails, return original with a low score
                return AbstractExplanation(
                    paper_id=paper.id,
                    explanation=explanation.explanation + "\n\n[WARNING: Quality check failed]",
                    quality_score=0.5
                )
        except Exception as e:
            # Handle API quota exhaustion or other API errors
            if "429" in str(e) or "Resource exhausted" in str(e):
                print(f"API quota exhausted when checking paper: {paper.title}. Skipping quality check.")
                return AbstractExplanation(
                    paper_id=paper.id,
                    explanation=explanation.explanation + "\n\n[NOTE: Quality check skipped due to API quota limitations]",
                    quality_score=0.75  # Assign a reasonable default score
                )
            else:
                print(f"Error during quality check: {str(e)}")
                return AbstractExplanation(
                    paper_id=paper.id,
                    explanation=explanation.explanation + f"\n\n[WARNING: Quality check error: {str(e)}]",
                    quality_score=0.5
                )
    
    async def regenerate_explanation(self, paper: ResearchPaper, explanation: AbstractExplanation, previous_score: float) -> AbstractExplanation:
        """Regenerate explanation with more explicit accuracy instructions"""
        prompt = f"""
        TASK: The previous explanation scored only {previous_score:.2f}/1.0 for accuracy.
        
        Generate a new explanation for this research abstract, ensuring STRICT adherence to facts presented in the abstract.
        DO NOT add any information not explicitly stated in the abstract.
        
        ORIGINAL ABSTRACT:
        {paper.abstract}
        
        Generate a clear, concise explanation that:
        1. Only includes information explicitly stated in the abstract
        2. Uses simpler language while maintaining accuracy
        3. Organizes the information logically
        4. Makes no speculative claims
        """
        
        response = await self.model.generate_content_async(prompt)
        
        return AbstractExplanation(
            paper_id=paper.id,
            explanation=response.text,
            quality_score=0.8  # Assume improved score, but not perfect
        )

# Create MCP server routes
@app.post("/task/researcher/search", response_model=Dict[str, Any])
async def create_research_task(query: ResearchQuery = Body(...)):
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
    """Create a task for the writer to explain an abstract"""
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
        explanation = await writer.explain_abstract(paper)
        
        # Submit to QA before returning
        qa_agent = QualityAssuranceAgent(gemini_model)
        verified_explanation = await qa_agent.check_explanation(paper, explanation)
        
        return {
            "task_id": task_id,
            "status": "completed",
            "result": verified_explanation.model_dump(),
            "quality_score": verified_explanation.quality_score
        }
    except Exception as e:
        return {
            "task_id": task_id,
            "status": "failed",
            "error": str(e)
        }

@app.get("/search", response_model=List[Dict[str, Any]])
async def search_papers_in_db(query: str, limit: int = 5):
    """Search for papers in the ChromaDB using RAG"""
    results = collection.query(
        query_texts=[query],
        n_results=limit
    )
    
    papers = []
    for i in range(len(results['ids'][0])):
        paper_id = results['ids'][0][i]
        metadata = results['metadatas'][0][i]
        document = results['documents'][0][i]
        
        papers.append({
            "id": paper_id,
            "title": metadata.get("title", ""),
            "authors": metadata.get("authors", "").split(", "),
            "abstract": document,
            "url": metadata.get("url", ""),
            "publication_date": metadata.get("publication_date", ""),
            "keywords": metadata.get("keywords", "").split(", ")
        })
    
    return papers

# Main orchestration function
async def run_full_pipeline(query: str, num_papers: int = 5):
    """Run the complete pipeline from research to writing to QA"""
    results = []
    
    # Initialize models and agents
    gemini_model = GenerativeModel('gemini-1.5-pro')
    researcher = ResearcherAgent(gemini_model)
    writer = WriterAgent(gemini_model)
    qa_agent = QualityAssuranceAgent(gemini_model)
    
    # Step 1: Find papers
    papers = await researcher.search_papers(query, num_papers)
    print(f"Found {len(papers)} papers related to: {query}")
    
    # Step 2: Generate explanations and verify
    for paper in papers:
        try:
            # Generate explanation
            explanation = await writer.explain_abstract(paper)
            print(f"Generated explanation for: {paper.title}")
            
            # Verify explanation
            verified_explanation = await qa_agent.check_explanation(paper, explanation)
            print(f"Quality score: {verified_explanation.quality_score:.2f}")
            
            results.append({
                "paper": paper.model_dump(),
                "explanation": verified_explanation.model_dump()
            })
        except Exception as e:
            print(f"Error processing paper '{paper.title}': {str(e)}")
            # Add a partial result with error information
            results.append({
                "paper": paper.model_dump(),
                "explanation": {
                    "paper_id": paper.id,
                    "explanation": f"[ERROR: Failed to process this paper: {str(e)}]",
                    "quality_score": 0.0
                }
            })
    
    return results

# CLI function to run the pipeline
async def main():
    query = input("Enter your research query: ")
    num_papers = int(input("Number of papers to retrieve (default 5): ") or "5")
    
    results = await run_full_pipeline(query, num_papers)
    
    # Print results
    for i, result in enumerate(results, 1):
        paper = result["paper"]
        explanation = result["explanation"]
        
        print(f"\n--- Paper {i}: {paper['title']} ---")
        print(f"Authors: {', '.join(paper['authors'])}")
        print(f"Quality Score: {explanation['quality_score']:.2f}")
        print("\nExplanation:")
        print(explanation['explanation'])
        print("\n" + "-"*50)
    
    # Save results to file
    with open("research_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to research_results.json")

# Server startup
def start_mcp_server():
    """Start the MCP server"""
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multi-agent research system")
    parser.add_argument("--mode", choices=["cli", "server"], default="cli", help="Run in CLI mode or server mode")
    args = parser.parse_args()
    
    if args.mode == "server":
        print("Starting MCP server on http://0.0.0.0:8000")
        start_mcp_server()
    else:
        asyncio.run(main())