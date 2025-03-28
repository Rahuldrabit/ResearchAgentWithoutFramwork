import time
from typing import List, Dict, Any, Optional
from google.generativeai import GenerativeModel
from tools.rate_limiter import rate_limiter
from tools.vector_db import retrieve_context_for_question

class ChatQAAgent:
    def __init__(self, gemini_model: GenerativeModel):
        self.model = gemini_model
        self.conversation_history = {}  # Store conversation history by session ID
    
    def _get_or_create_history(self, session_id: str) -> List[Dict[str, str]]:
        """Get existing conversation history or create a new one"""
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = [
                {"role": "system", "content": "You are a helpful research assistant that answers questions based on the provided research paper information. When answering questions, use the context provided, and if you don't know the answer, say so. Always cite the sources of your information."}
            ]
        return self.conversation_history[session_id]
    
    async def answer_question(self, 
                             question: str, 
                             session_id: Optional[str] = None,
                             max_context_papers: int = 3) -> Dict[str, Any]:
        """
        Answer a question using RAG approach with paper database
        
        Args:
            question: The user's question
            session_id: Optional conversation session ID for continuity
            max_context_papers: Maximum number of papers to include as context
            
        Returns:
            Dict with answer and sources
        """
        start_time = time.time()
        
        # Create or get conversation history
        session_id = session_id or str(int(time.time()))
        history = self._get_or_create_history(session_id)
        
        # Retrieve relevant papers from vector DB
        relevant_papers = retrieve_context_for_question(
            question=question, 
            limit=max_context_papers
        )
        
        # Prepare context from papers
        contexts = []
        used_sources = []
        
        for i, paper in enumerate(relevant_papers):
            context_text = f"Paper {i+1}: {paper['title']}\nAuthors: {', '.join(paper['authors'])}\nAbstract: {paper['abstract']}\n"
            contexts.append(context_text)
            
            # Track sources for citation
            used_sources.append({
                "id": paper["id"],
                "title": paper["title"],
                "authors": paper["authors"],
                "relevance_score": paper["relevance_score"],
                "url": paper["url"]
            })
        
        # Prepare the prompt with all context
        context_text = "\n\n".join(contexts)
        
        # Add user question to history
        history.append({"role": "user", "content": question})
        
        # Construct the RAG prompt
        rag_prompt = f"""
        Based on the following research papers, please answer this question: {question}
        
        RESEARCH PAPERS CONTEXT:
        {context_text}
        
        Please provide a comprehensive answer that:
        1. Directly addresses the question
        2. Cites specific papers from the context when providing information
        3. Mentions if information is not available in the provided context
        4. Synthesizes information across papers when applicable
        
        If the context doesn't contain relevant information, please indicate that you cannot answer based on the available information.
        """
        
        try:
            # Get answer from LLM
            response = await rate_limiter.execute_with_retry(
                self.model.generate_content_async,
                rag_prompt
            )
            
            # Add assistant response to history (truncate history if too long)
            history.append({"role": "assistant", "content": response.text})
            
            # Keep history to reasonable size (last 10 messages)
            if len(history) > 12:  # system message + 10 exchanges
                history = [history[0]] + history[-10:]
            
            self.conversation_history[session_id] = history
            
            return {
                "answer": response.text,
                "sources": used_sources,
                "session_id": session_id,
                "processing_time": time.time() - start_time,
                "context_papers_count": len(relevant_papers)
            }
            
        except Exception as e:
            print(f"Error in ChatQA: {str(e)}")
            return {
                "answer": f"I encountered an error while trying to answer your question: {str(e)}",
                "sources": [],
                "session_id": session_id,
                "error": str(e)
            }
    
    def clear_conversation(self, session_id: str) -> bool:
        """Clear a specific conversation history"""
        if session_id in self.conversation_history:
            # Keep the system message
            self.conversation_history[session_id] = self.conversation_history[session_id][:1]
            return True
        return False
