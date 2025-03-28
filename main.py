import asyncio
import argparse
import json
import time
from google.generativeai import GenerativeModel
import google.generativeai as genai

from config import GOOGLE_API_KEY
from models.data_models import EnhancedAbstractExplanation
from agents.researcher import ResearcherAgent
from agents.writer import WriterAgent
from agents.quality_assurance import EnhancedQualityAssuranceAgent
from agents.chat_qa import ChatQAAgent
from tools.rate_limiter import rate_limiter
from api.server import start_server

# Configure Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

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

async def main_cli():
    """CLI interface for the application"""
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

async def chat_mode():
    """Interactive chat mode with the research database"""
    print("\n===== Research Assistant Chat Mode =====")
    print("Ask questions about research papers. Type 'exit' to quit.\n")
    
    gemini_model = GenerativeModel('gemini-1.5-pro')
    chat_agent = ChatQAAgent(gemini_model)
    session_id = str(int(time.time()))
    
    while True:
        question = input("\nYour question: ")
        if question.lower() in ['exit', 'quit', 'bye']:
            print("Goodbye!")
            break
            
        print("\nSearching database and generating answer...")
        result = await chat_agent.answer_question(
            question=question,
            session_id=session_id
        )
        
        print("\n" + "=" * 80)
        print(result['answer'])
        print("=" * 80)
        
        if result['sources']:
            print("\nSources:")
            for i, source in enumerate(result['sources']):
                print(f"{i+1}. {source['title']} by {', '.join(source['authors'][:2])}" + 
                     (f" et al." if len(source['authors']) > 2 else ""))
                print(f"   Relevance: {source['relevance_score']:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-agent research system")
    parser.add_argument("--mode", choices=["cli", "server", "chat"], default="cli", 
                      help="Run in CLI mode, server mode, or chat mode")
    parser.add_argument("--max-rpm", type=int, default=5, 
                      help="Maximum API requests per minute (default: 10)")
    args = parser.parse_args()
    
    # Configure rate limiter based on CLI arguments
    rate_limiter.max_calls_per_minute = args.max_rpm
    
    if args.mode == "server":
        print(f"Starting MCP server on http://0.0.0.0:8000 (API rate limit: {args.max_rpm} requests/min)")
        start_server()
    elif args.mode == "chat":
        asyncio.run(chat_mode())
    else:
        asyncio.run(main_cli())
