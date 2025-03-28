import uuid
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

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
