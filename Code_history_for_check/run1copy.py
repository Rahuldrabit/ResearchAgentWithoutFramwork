import os
import json
import uuid
import time
import random
from typing import List, Dict, Any, Optional

# For Gemini API
from google.generativeai import GenerativeModel
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable

# For vector database
import chromadb
from chromadb.utils import embedding_functions

# For MCP server
from fastapi import FastAPI, Body, HTTPException, BackgroundTasks
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
    
    def model_dump(self):
        """Compatibility method for both Pydantic v1 and v2"""
        if hasattr(super(), "model_dump"):
            return super().model_dump()
        return self.dict()

class AbstractExplanation(BaseModel):
    paper_id: str
    explanation: str
    quality_score: float = 0.0
    
    def model_dump(self):
        """Compatibility method for both Pydantic v1 and v2"""
        if hasattr(super(), "model_dump"):
            return super().model_dump()
        return self.dict()

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
    
    def model_dump(self):
        """Compatibility method for both Pydantic v1 and v2"""
        if hasattr(super(), "model_dump"):
            return super().model_dump()
        return self.dict()

# Rate limiting utility
class RateLimiter:
    def __init__(self, max_calls_per_minute=10, max_retries=5, initial_delay=2):
        self.max_calls_per_minute = max_calls_per_minute
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.calls = []
    
    async def wait_if_needed(self):
        """Wait if we've made too many calls in the last minute"""
        now = time.time()
        # Remove calls older than 1 minute
        self.calls = [t for t in self.calls if now - t < 60]
        
        if len(self.calls) >= self.max_calls_per_minute:
            wait_time = 60 - (now - self.calls[0]) + 1  # Add 1 second buffer
            print(f"Rate limit reached. Waiting {wait_time:.2f} seconds...")
            await asyncio.sleep(wait_time)
        
        # Add this call
        self.calls.append(time.time())
    
    async def execute_with_retry(self, func, *args, **kwargs):
        """Execute a function with exponential backoff retry logic"""
        for attempt in range(self.max_retries):
            try:
                await self.wait_if_needed()
                return await func(*args, **kwargs)
            except (ResourceExhausted, ServiceUnavailable) as e:
                delay = self.initial_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"API quota exceeded. Retrying in {delay:.2f} seconds... (Attempt {attempt+1}/{self.max_retries})")
                await asyncio.sleep(delay)
        
        # If we get here, all retries failed
        raise Exception(f"Failed after {self.max_retries} attempts due to API quota limits")

# Create rate limiter for the entire application
rate_limiter = RateLimiter(max_calls_per_minute=10)

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
        
        response = await rate_limiter.execute_with_retry(
            self.model.generate_content_async,
            search_prompt
        )
        
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
        
        response = await rate_limiter.execute_with_retry(
            self.model.generate_content_async,
            prompt
        )
        
        return AbstractExplanation(
            paper_id=paper.id,
            explanation=response.text
        )
class EnhancedQualityAssuranceAgent:
    def __init__(self, gemini_model: GenerativeModel):
        self.model = gemini_model
        self.evaluation_history = {}  # Track history for continuous improvement
    
    async def evaluate_explanation(self, paper: ResearchPaper, explanation: AbstractExplanation) -> AbstractExplanation:
        """Comprehensive evaluation of explanation quality against multiple metrics"""
        
        # Define evaluation criteria
        prompt = f"""
        TASK: Perform a comprehensive evaluation of the explanation below against the original abstract.
        
        ORIGINAL ABSTRACT:
        {paper.abstract}
        
        EXPLANATION:
        {explanation.explanation}
        
        Evaluate based on the following criteria:
        
        1. FACTUAL ACCURACY (0-10):
           - Are all claims in the explanation supported by the abstract?
           - Are there any hallucinations or invented details?
           - Are methodologies and findings accurately represented?
        
        2. COMPREHENSIVENESS (0-10):
           - Does the explanation cover all key points from the abstract?
           - Are important findings or methodologies omitted?
        
        3. CLARITY (0-10):
           - Is the explanation clear and understandable to a non-expert?
           - Are technical terms adequately explained?
           - Is the writing well-structured and logical?
        
        4. CONCISENESS (0-10):
           - Is the explanation appropriately concise without sacrificing important details?
           - Is there unnecessary repetition or verbosity?
        
        For each criterion, provide:
        - A score (0-10)
        - Specific examples supporting your assessment
        - Suggested improvements
        
        Also identify any specific hallucinations or factual errors with explanation.
        
        Return your evaluation as a JSON object with the following structure:
        {
            "factual_accuracy": {
                "score": [0-10],
                "issues": ["issue1", "issue2", ...],
                "suggestions": ["suggestion1", "suggestion2", ...]
            },
            "comprehensiveness": {
                "score": [0-10],
                "issues": ["issue1", "issue2", ...],
                "suggestions": ["suggestion1", "suggestion2", ...]
            },
            "clarity": {
                "score": [0-10],
                "issues": ["issue1", "issue2", ...],
                "suggestions": ["suggestion1", "suggestion2", ...]
            },
            "conciseness": {
                "score": [0-10],
                "issues": ["issue1", "issue2", ...],
                "suggestions": ["suggestion1", "suggestion2", ...]
            },
            "hallucinations": ["specific hallucination 1", "specific hallucination 2", ...],
            "overall_quality_score": [0.0-1.0],
            "corrected_explanation": "improved version that fixes critical issues"
        }
        """
        
        try:
            response = await rate_limiter.execute_with_retry(
                self.model.generate_content_async,
                prompt
            )
            
            # Extract JSON from response
            response_text = response.text
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            evaluation = json.loads(response_text)
            
            # Calculate weighted quality score if not provided or invalid
            if not isinstance(evaluation.get("overall_quality_score"), (int, float)) or not 0 <= evaluation.get("overall_quality_score") <= 1:
                # Factual accuracy should be weighted most heavily
                weights = {"factual_accuracy": 0.5, "comprehensiveness": 0.2, "clarity": 0.15, "conciseness": 0.15}
                weighted_score = sum(
                    weights[metric] * (evaluation.get(metric, {}).get("score", 0) / 10)
                    for metric in weights
                )
                evaluation["overall_quality_score"] = min(max(weighted_score, 0.0), 1.0)
            
            # Store evaluation in history for this paper
            if paper.id not in self.evaluation_history:
                self.evaluation_history[paper.id] = []
            self.evaluation_history[paper.id].append({
                "explanation_text": explanation.explanation,
                "evaluation": evaluation,
                "timestamp": time.time()
            })
            
            # Create enhanced explanation result
            corrected_explanation = evaluation.get("corrected_explanation", explanation.explanation)
            quality_score = evaluation.get("overall_quality_score", 0.0)
            
            # Add metadata about the evaluation
            issues_summary = []
            for category in ["factual_accuracy", "comprehensiveness", "clarity", "conciseness"]:
                cat_data = evaluation.get(category, {})
                if "issues" in cat_data and cat_data["issues"]:
                    issues_summary.extend(cat_data["issues"])
            
            metadata = {
                "evaluation_metrics": {
                    "factual_accuracy": evaluation.get("factual_accuracy", {}).get("score", 0),
                    "comprehensiveness": evaluation.get("comprehensiveness", {}).get("score", 0),
                    "clarity": evaluation.get("clarity", {}).get("score", 0),
                    "conciseness": evaluation.get("conciseness", {}).get("score", 0)
                },
                "hallucinations": evaluation.get("hallucinations", []),
                "issues": issues_summary[:3]  # Limit to top 3 issues
            }
            
            # Decision logic for regeneration
            if quality_score < 0.7 or evaluation.get("factual_accuracy", {}).get("score", 0) < 7:
                return await self.intelligent_regeneration(paper, explanation, evaluation)
            
            result = AbstractExplanation(
                paper_id=paper.id,
                explanation=corrected_explanation,
                quality_score=quality_score
            )
            # Attach metadata (this would require extending the AbstractExplanation class)
            # For now we'll just return the basic version
            return result
            
        except Exception as e:
            print(f"Enhanced QA evaluation failed: {str(e)}")
            # If parsing fails, return original with a warning
            return AbstractExplanation(
                paper_id=paper.id,
                explanation=explanation.explanation + "\n\n[WARNING: Quality evaluation failed. Exercise caution when using this explanation.]",
                quality_score=0.5
            )
    
    async def intelligent_regeneration(self, paper: ResearchPaper, explanation: AbstractExplanation, evaluation: dict) -> AbstractExplanation:
        """Targeted regeneration based on specific issues identified"""
        
        # Extract specific issues for targeted improvement
        factual_issues = evaluation.get("factual_accuracy", {}).get("issues", [])
        factual_suggestions = evaluation.get("factual_accuracy", {}).get("suggestions", [])
        hallucinations = evaluation.get("hallucinations", [])
        
        # Determine most critical areas for improvement
        critical_issues = []
        if hallucinations:
            critical_issues.append("Hallucinations detected: " + "; ".join(hallucinations[:3]))
        if factual_issues:
            critical_issues.append("Factual issues: " + "; ".join(factual_issues[:3]))
        
        # Build an improvement-focused prompt
        prompt = f"""
        TASK: Generate an improved explanation for this research abstract.
        
        ORIGINAL ABSTRACT:
        {paper.abstract}
        
        PREVIOUS EXPLANATION:
        {explanation.explanation}
        
        CRITICAL ISSUES TO FIX:
        {chr(10).join([f"- {issue}" for issue in critical_issues])}
        
        IMPROVEMENT SUGGESTIONS:
        {chr(10).join([f"- {suggestion}" for suggestion in factual_suggestions])}
        
        REQUIREMENTS:
        1. Create a new explanation that addresses ALL the issues above
        2. Ensure STRICT factual accuracy - only include information from the abstract
        3. Be clear and concise while maintaining comprehensive coverage
        4. Format in clear paragraphs with a logical structure
        5. DO NOT add any speculative information or details not in the abstract
        6. DO NOT use phrases like "according to the abstract" or "the authors state" - just present the information clearly
        
        Your response should contain ONLY the improved explanation text.
        """
        
        try:
            response = await rate_limiter.execute_with_retry(
                self.model.generate_content_async,
                prompt
            )
            
            # Quick verification check on the regenerated content
            verification_prompt = f"""
            VERIFICATION TASK: Does this explanation contain ANY hallucinations or information not present in the original abstract?
            
            ORIGINAL ABSTRACT:
            {paper.abstract}
            
            REGENERATED EXPLANATION:
            {response.text}
            
            Answer ONLY with "YES" if there are hallucinations or "NO" if the explanation is strictly factual based on the abstract.
            """
            
            verification = await rate_limiter.execute_with_retry(
                self.model.generate_content_async,
                verification_prompt
            )
            
            verified = verification.text.strip().upper().startswith("NO")
            
            if verified:
                return AbstractExplanation(
                    paper_id=paper.id,
                    explanation=response.text,
                    quality_score=0.85  # Higher score for verified regeneration
                )
            else:
                # If still not verified, try one more time with stricter instructions
                return await self.final_fallback_regeneration(paper)
                
        except Exception as e:
            print(f"Intelligent regeneration failed: {str(e)}")
            # If regeneration fails completely, use a more conservative approach
            return await self.final_fallback_regeneration(paper)
    
    async def final_fallback_regeneration(self, paper: ResearchPaper) -> AbstractExplanation:
        """Ultra-conservative regeneration that prioritizes accuracy above all else"""
        
        ultra_safe_prompt = f"""
        TASK: Create an extremely factual explanation of this research abstract.
        
        ABSTRACT:
        {paper.abstract}
        
        CRITICAL REQUIREMENTS:
        1. ONLY include information explicitly stated in the abstract
        2. Use simple, clear language
        3. Organize information in a logical structure
        4. If you're uncertain about ANY detail, omit it entirely
        5. Focus on the main research question, methodology, and key findings
        6. Keep it concise and focused
        
        Your response should be a conservative, factually accurate explanation of the research.
        """
        
        try:
            response = await rate_limiter.execute_with_retry(
                self.model.generate_content_async,
                ultra_safe_prompt
            )
            
            return AbstractExplanation(
                paper_id=paper.id,
                explanation=response.text + "\n\n[Note: This explanation has been regenerated with a focus on strict factual accuracy.]",
                quality_score=0.75  # Reasonable score for conservative approach
            )
        except Exception as e:
            print(f"Final fallback regeneration failed: {str(e)}")
            # Absolute last resort
            return AbstractExplanation(
                paper_id=paper.id,
                explanation="Unable to generate a reliable explanation for this research paper. Please refer to the original abstract.",
                quality_score=0.0
            )
    
    async def bulk_evaluation(self, papers_and_explanations: List[tuple]) -> List[AbstractExplanation]:
        """Process multiple papers and explanations with rate limiting"""
        results = []
        
        for paper, explanation in papers_and_explanations:
            try:
                verified_explanation = await self.evaluate_explanation(paper, explanation)
                results.append(verified_explanation)
                # Sleep briefly to avoid overwhelming the API
                await asyncio.sleep(0.5)
            except Exception as e:
                print(f"Error processing {paper.title}: {str(e)}")
                results.append(AbstractExplanation(
                    paper_id=paper.id,
                    explanation=explanation.explanation + "\n\n[ERROR: Quality check failed]",
                    quality_score=0.0
                ))
        
        return results
    
    def get_improvement_insights(self, paper_id: str = None) -> Dict:
        """Analyze evaluation history to identify common issues and improvement patterns"""
        if paper_id and paper_id in self.evaluation_history:
            # Analyze specific paper history
            history = self.evaluation_history[paper_id]
        else:
            # Aggregate all history
            history = []
            for paper_history in self.evaluation_history.values():
                history.extend(paper_history)
        
        if not history:
            return {"error": "No evaluation history available"}
        
        # Aggregate metrics
        metrics = {
            "factual_accuracy": [],
            "comprehensiveness": [],
            "clarity": [],
            "conciseness": [],
            "overall_quality": []
        }
        
        common_issues = {}
        hallucinations = []
        
        for entry in history:
            eval_data = entry.get("evaluation", {})
            
            # Collect metrics
            metrics["overall_quality"].append(eval_data.get("overall_quality_score", 0))
            
            for metric in ["factual_accuracy", "comprehensiveness", "clarity", "conciseness"]:
                if metric in eval_data and "score" in eval_data[metric]:
                    metrics[metric].append(eval_data[metric]["score"])
            
            # Collect issues
            for metric in ["factual_accuracy", "comprehensiveness", "clarity", "conciseness"]:
                if metric in eval_data and "issues" in eval_data[metric]:
                    for issue in eval_data[metric]["issues"]:
                        if issue in common_issues:
                            common_issues[issue] += 1
                        else:
                            common_issues[issue] = 1
            
            # Collect hallucinations
            if "hallucinations" in eval_data:
                hallucinations.extend(eval_data["hallucinations"])
        
        # Calculate average metrics
        avg_metrics = {k: sum(v)/len(v) if v else 0 for k, v in metrics.items()}
        
        # Sort issues by frequency
        sorted_issues = sorted(common_issues.items(), key=lambda x: x[1], reverse=True)
        top_issues = [{"issue": k, "count": v} for k, v in sorted_issues[:10]]
        
        # Get most common hallucinations
        hallucination_counts = {}
        for h in hallucinations:
            if h in hallucination_counts:
                hallucination_counts[h] += 1
            else:
                hallucination_counts[h] = 1
        
        sorted_hallucinations = sorted(hallucination_counts.items(), key=lambda x: x[1], reverse=True)
        top_hallucinations = [{"hallucination": k, "count": v} for k, v in sorted_hallucinations[:10]]
        
        return {
            "average_metrics": avg_metrics,
            "top_issues": top_issues,
            "top_hallucinations": top_hallucinations,
            "total_evaluations": len(history),
            "improvement_suggestions": self._generate_improvement_suggestions(avg_metrics, top_issues)
        }
    
    def _generate_improvement_suggestions(self, metrics, issues):
        """Generate improvement suggestions based on metrics and common issues"""
        suggestions = []
        
        # Check factual accuracy
        if metrics["factual_accuracy"] < 7:
            suggestions.append("Focus on improving factual accuracy by implementing stricter verification")
        
        # Check comprehensiveness
        if metrics["comprehensiveness"] < 7:
            suggestions.append("Ensure all key points from the abstract are covered in explanations")
        
        # Check clarity
        if metrics["clarity"] < 7:
            suggestions.append("Improve clarity by using simpler language and better structure")
        
        # Check conciseness
        if metrics["conciseness"] < 7:
            suggestions.append("Make explanations more concise by eliminating redundancy")
        
        # Add specific suggestions based on common issues
        if issues:
            suggestions.append(f"Address common issue: {issues[0]['issue']}")
            if len(issues) > 1:
                suggestions.append(f"Address common issue: {issues[1]['issue']}")
        
        return suggestions
    
    
# Updated server routes using the enhanced QA agent
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

# Add a new endpoint for batch processing with QA
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

# Add a QA-specific analytics endpoint
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
    
# Create MCP server routes
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

class EnhancedAbstractExplanation(BaseModel):
    paper_id: str
    explanation: str
    quality_score: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def model_dump(self):
        """Compatibility method for both Pydantic v1 and v2"""
        if hasattr(super(), "model_dump"):
            return super().model_dump()
        return self.dict()
    
    @classmethod
    def from_basic(cls, basic_explanation: AbstractExplanation, metadata: Dict[str, Any] = None):
        """Convert a basic AbstractExplanation to an enhanced one"""
        return cls(
            paper_id=basic_explanation.paper_id,
            explanation=basic_explanation.explanation,
            quality_score=basic_explanation.quality_score,
            metadata=metadata or {}
        )
    
    def add_metadata(self, key: str, value: Any):
        """Add metadata to the explanation"""
        self.metadata[key] = value
        return self
    
    def with_annotation(self, annotation: str) -> 'EnhancedAbstractExplanation':
        """Add an annotation to the explanation text"""
        return EnhancedAbstractExplanation(
            paper_id=self.paper_id,
            explanation=f"{self.explanation}\n\n[Note: {annotation}]",
            quality_score=self.quality_score,
            metadata=self.metadata
        )
    
    def get_evaluation_summary(self) -> str:
        """Generate a human-readable summary of the evaluation"""
        if not self.metadata or "evaluation_metrics" not in self.metadata:
            return "No evaluation data available."
        
        metrics = self.metadata.get("evaluation_metrics", {})
        issues = self.metadata.get("issues", [])
        hallucinations = self.metadata.get("hallucinations", [])
        
        summary = [
            f"Quality Score: {self.quality_score:.2f}/1.0",
            f"Factual Accuracy: {metrics.get('factual_accuracy', 0)}/10",
            f"Comprehensiveness: {metrics.get('comprehensiveness', 0)}/10",
            f"Clarity: {metrics.get('clarity', 0)}/10",
            f"Conciseness: {metrics.get('conciseness', 0)}/10",
        ]
        
        if issues:
            summary.append("\nKey Issues:")
            for issue in issues:
                summary.append(f"- {issue}")
        
        if hallucinations:
            summary.append("\nPotential Hallucinations:")
            for hall in hallucinations:
                summary.append(f"- {hall}")
        
        return "\n".join(summary)

# Main orchestration function
async def run_full_pipeline(query: str, num_papers: int = 5):
    """Run the complete pipeline from research to writing to QA"""
    results = []
    
    # Initialize models and agents
    gemini_model = GenerativeModel('gemini-1.5-pro')
    researcher = ResearcherAgent(gemini_model)
    writer = WriterAgent(gemini_model)
    qa_agent = EnhancedQualityAssuranceAgent(gemini_model)
    
    # Step 1: Find papers
    try:
        papers = await researcher.search_papers(query, num_papers)
        print(f"Found {len(papers)} papers related to: {query}")
        
        # Step 2: Generate explanations and verify
        for paper in papers:
            try:
                # Generate explanation
                explanation = await writer.explain_abstract(paper)
                print(f"Generated explanation for: {paper.title}")
                
                # Verify explanation with enhanced QA
                try:
                    verified_explanation = await qa_agent.evaluate_explanation(paper, explanation)
                    print(f"Quality score: {verified_explanation.quality_score:.2f}")
                    
                    # Create enhanced explanation with metadata
                    enhanced_explanation = EnhancedAbstractExplanation.from_basic(
                        verified_explanation,
                        metadata=getattr(verified_explanation, 'metadata', {})
                    )
                    
                    # Add evaluation summary
                    print(enhanced_explanation.get_evaluation_summary())
                    
                except Exception as e:
                    print(f"QA check failed for {paper.title}: {str(e)}")
                    # Use original explanation if QA fails
                    enhanced_explanation = EnhancedAbstractExplanation(
                        paper_id=paper.id,
                        explanation=explanation.explanation + "\n\n[WARNING: Quality check failed]",
                        quality_score=0.5,
                        metadata={"error": str(e)}
                    )
                
                results.append({
                    "paper": paper.model_dump(),
                    "explanation": enhanced_explanation.model_dump()
                })
            except Exception as e:
                print(f"Failed to process paper {paper.title}: {str(e)}")
                # Add error entry for this paper
                results.append({
                    "paper": paper.model_dump(),
                    "explanation": {
                        "paper_id": paper.id,
                        "explanation": f"Error generating explanation: {str(e)}",
                        "quality_score": 0.0,
                        "metadata": {"error": str(e)}
                    }
                })
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        raise
    
    return results

# CLI function to run the pipeline with improved output formatting
async def main():
    query = input("Enter your research query: ")
    num_papers = int(input("Number of papers to retrieve (default 5): ") or "5")
    
    print("\nRunning research pipeline with rate limiting... This may take some time.")
    print("Each paper will be processed with caution to avoid API rate limits.\n")
    
    try:
        results = await run_full_pipeline(query, num_papers)
        
        # Print results with enhanced formatting
        for i, result in enumerate(results, 1):
            paper = result["paper"]
            explanation = result["explanation"]
            
            print(f"\n{'='*30} PAPER {i} {'='*30}")
            print(f"Title: {paper['title']}")
            print(f"Authors: {', '.join(paper['authors'])}")
            print(f"Keywords: {', '.join(paper.get('keywords', []))}")
            print(f"\nQuality Score: {explanation.get('quality_score', 0):.2f}/1.0")
            
            # Print evaluation metrics if available
            metadata = explanation.get('metadata', {})
            if metadata and 'evaluation_metrics' in metadata:
                metrics = metadata['evaluation_metrics']
                print("\nEvaluation Metrics:")
                for metric, score in metrics.items():
                    print(f"- {metric.replace('_', ' ').title()}: {score}/10")
            
            print("\nEXPLANATION:")
            print("-" * 70)
            print(explanation['explanation'])
            print("-" * 70)
            
            # Print issues if available
            if metadata and 'issues' in metadata and metadata['issues']:
                print("\nIdentified Issues:")
                for issue in metadata['issues']:
                    print(f"- {issue}")
            
            # Print hallucinations if available
            if metadata and 'hallucinations' in metadata and metadata['hallucinations']:
                print("\nPotential Hallucinations:")
                for hall in metadata['hallucinations']:
                    print(f"- {hall}")
            
            print("=" * 70)
        
        # Save results to file
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"research_results_{timestamp}.json"
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {filename}")
    except Exception as e:
        print(f"Error in main pipeline: {str(e)}")
        print("\nPartial results may be available. Check the console output.")

# Server startup
def start_mcp_server():
    """Start the MCP server"""
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multi-agent research system")
    parser.add_argument("--mode", choices=["cli", "server"], default="cli", help="Run in CLI mode or server mode")
    parser.add_argument("--max-rpm", type=int, default=10, help="Maximum API requests per minute (default: 10)")
    args = parser.parse_args()  # Fix: Parse the arguments from the parser object
    
    # Configure rate limiter based on CLI arguments
    rate_limiter.max_calls_per_minute = args.max_rpm
    
    if args.mode == "server":
        print(f"Starting MCP server on http://0.0.0.0:8000 (API rate limit: {args.max_rpm} requests/min)")
        start_mcp_server()
    else:
        asyncio.run(main())

# Add a simple QualityAssuranceAgent class for backward compatibility
class QualityAssuranceAgent:
    def __init__(self, gemini_model: GenerativeModel):
        self.model = gemini_model
        # Create an enhanced agent to delegate tasks to
        self.enhanced_agent = EnhancedQualityAssuranceAgent(gemini_model)
    
    async def check_explanation(self, paper: ResearchPaper, explanation: AbstractExplanation) -> AbstractExplanation:
        """Delegate to enhanced agent for better quality checks"""
        return await self.enhanced_agent.evaluate_explanation(paper, explanation)

# Update server routes to use the enhanced QA agent
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

# Update the main orchestration function
async def run_full_pipeline(query: str, num_papers: int = 5):
    """Run the complete pipeline from research to writing to QA"""
    results = []
    
    # Initialize models and agents
    gemini_model = GenerativeModel('gemini-1.5-pro')
    researcher = ResearcherAgent(gemini_model)
    writer = WriterAgent(gemini_model)
    qa_agent = EnhancedQualityAssuranceAgent(gemini_model)
    
    # Step 1: Find papers
    try:
        papers = await researcher.search_papers(query, num_papers)
        print(f"Found {len(papers)} papers related to: {query}")
        
        # Step 2: Generate explanations and verify
        for paper in papers:
            try:
                # Generate explanation
                explanation = await writer.explain_abstract(paper)
                print(f"Generated explanation for: {paper.title}")
                
                # Verify explanation with enhanced QA
                try:
                    verified_explanation = await qa_agent.evaluate_explanation(paper, explanation)
                    print(f"Quality score: {verified_explanation.quality_score:.2f}")
                    
                    # Create enhanced explanation with metadata
                    enhanced_explanation = EnhancedAbstractExplanation.from_basic(
                        verified_explanation,
                        metadata=getattr(verified_explanation, 'metadata', {})
                    )
                    
                    # Add evaluation summary
                    print(enhanced_explanation.get_evaluation_summary())
                    
                except Exception as e:
                    print(f"QA check failed for {paper.title}: {str(e)}")
                    # Use original explanation if QA fails
                    enhanced_explanation = EnhancedAbstractExplanation(
                        paper_id=paper.id,
                        explanation=explanation.explanation + "\n\n[WARNING: Quality check failed]",
                        quality_score=0.5,
                        metadata={"error": str(e)}
                    )
                
                results.append({
                    "paper": paper.model_dump(),
                    "explanation": enhanced_explanation.model_dump()
                })
            except Exception as e:
                print(f"Failed to process paper {paper.title}: {str(e)}")
                # Add error entry for this paper
                results.append({
                    "paper": paper.model_dump(),
                    "explanation": {
                        "paper_id": paper.id,
                        "explanation": f"Error generating explanation: {str(e)}",
                        "quality_score": 0.0,
                        "metadata": {"error": str(e)}
                    }
                })
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        raise
    
    return results

# CLI function to run the pipeline with improved output formatting
async def main():
    query = input("Enter your research query: ")
    num_papers = int(input("Number of papers to retrieve (default 5): ") or "5")
    
    print("\nRunning research pipeline with rate limiting... This may take some time.")
    print("Each paper will be processed with caution to avoid API rate limits.\n")
    
    try:
        results = await run_full_pipeline(query, num_papers)
        
        # Print results with enhanced formatting
        for i, result in enumerate(results, 1):
            paper = result["paper"]
            explanation = result["explanation"]
            
            print(f"\n{'='*30} PAPER {i} {'='*30}")
            print(f"Title: {paper['title']}")
            print(f"Authors: {', '.join(paper['authors'])}")
            print(f"Keywords: {', '.join(paper.get('keywords', []))}")
            print(f"\nQuality Score: {explanation.get('quality_score', 0):.2f}/1.0")
            
            # Print evaluation metrics if available
            metadata = explanation.get('metadata', {})
            if metadata and 'evaluation_metrics' in metadata:
                metrics = metadata['evaluation_metrics']
                print("\nEvaluation Metrics:")
                for metric, score in metrics.items():
                    print(f"- {metric.replace('_', ' ').title()}: {score}/10")
            
            print("\nEXPLANATION:")
            print("-" * 70)
            print(explanation['explanation'])
            print("-" * 70)
            
            # Print issues if available
            if metadata and 'issues' in metadata and metadata['issues']:
                print("\nIdentified Issues:")
                for issue in metadata['issues']:
                    print(f"- {issue}")
            
            # Print hallucinations if available
            if metadata and 'hallucinations' in metadata and metadata['hallucinations']:
                print("\nPotential Hallucinations:")
                for hall in metadata['hallucinations']:
                    print(f"- {hall}")
            
            print("=" * 70)
        
        # Save results to file
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"research_results_{timestamp}.json"
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {filename}")
    except Exception as e:
        print(f"Error in main pipeline: {str(e)}")
        print("\nPartial results may be available. Check the console output.")
        