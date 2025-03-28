import json
from typing import List
from google.generativeai import GenerativeModel
from models.data_models import ResearchPaper
from tools.rate_limiter import rate_limiter
from tools.vector_db import store_paper

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
                
                # Store in vector database
                store_paper(paper)
            
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
