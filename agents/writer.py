from google.generativeai import GenerativeModel
from models.data_models import ResearchPaper, AbstractExplanation
from tools.rate_limiter import rate_limiter

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
