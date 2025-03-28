import json
import time
import asyncio
from typing import List, Dict, Any, Tuple
from google.generativeai import GenerativeModel
from models.data_models import ResearchPaper, AbstractExplanation, EnhancedAbstractExplanation
from tools.rate_limiter import rate_limiter

class QualityAssuranceAgent:
    def __init__(self, gemini_model: GenerativeModel):
        self.model = gemini_model
        # Create an enhanced agent to delegate tasks to
        self.enhanced_agent = EnhancedQualityAssuranceAgent(gemini_model)
    
    async def check_explanation(self, paper: ResearchPaper, explanation: AbstractExplanation) -> AbstractExplanation:
        """Delegate to enhanced agent for better quality checks"""
        return await self.enhanced_agent.evaluate_explanation(paper, explanation)

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
    
    async def bulk_evaluation(self, papers_and_explanations: List[Tuple[ResearchPaper, AbstractExplanation]]) -> List[AbstractExplanation]:
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