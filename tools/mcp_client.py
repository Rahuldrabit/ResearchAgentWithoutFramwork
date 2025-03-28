import requests
import json
from typing import Dict, List, Any, Optional, Union
import os

class MCPClient:
    """
    Client for interacting with the Multi-agent Coordination Protocol (MCP) server.
    Provides methods to access all server endpoints including the enhanced QA functionality.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the MCP client.
        
        Args:
            base_url: The base URL of the MCP server. Defaults to http://localhost:8000.
        """
        self.base_url = base_url
        
    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None, 
                     params: Optional[Dict] = None, files: Optional[Dict] = None) -> Dict:
        """
        Make an HTTP request to the MCP server.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint to call
            data: JSON data to send in request body
            params: URL parameters
            files: Files to upload
            
        Returns:
            The JSON response from the server
        """
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method == "GET":
                response = requests.get(url, params=params)
            elif method == "POST":
                if files:
                    response = requests.post(url, data=data, files=files)
                else:
                    response = requests.post(url, json=data, params=params)
            elif method == "PUT":
                response = requests.put(url, json=data, params=params)
            elif method == "DELETE":
                response = requests.delete(url, params=params)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            
            if response.text:
                return response.json()
            return {}
            
        except requests.exceptions.RequestException as e:
            print(f"Error making request to {url}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response status code: {e.response.status_code}")
                print(f"Response text: {e.response.text}")
            raise
    
    # Research papers and search
    def search_papers(self, query: str, limit: int = 5) -> List[Dict]:
        """Search for papers in the database."""
        params = {"query": query, "limit": limit}
        return self._make_request("GET", "/search", params=params)
    
    def create_research_task(self, query: str, num_papers: int = 5) -> Dict:
        """Create a task to search for papers."""
        data = {"query": query, "num_papers": num_papers}
        return self._make_request("POST", "/task/researcher/search", data=data)
    
    def explain_paper(self, paper: Dict) -> Dict:
        """Create a task for the writer to explain a paper abstract."""
        return self._make_request("POST", "/task/writer/explain", data=paper)
    
    def batch_process_papers(self, papers: List[Dict]) -> Dict:
        """Process multiple papers with writing and QA in parallel."""
        return self._make_request("POST", "/batch/process", data=papers)
    
    # QA Analytics
    def get_qa_analytics(self, paper_id: Optional[str] = None) -> Dict:
        """Get analytics about QA performance."""
        params = {}
        if paper_id:
            params["paper_id"] = paper_id
        return self._make_request("GET", "/qa/analytics", params=params)
    
    # Chat functionality
    def ask_chat_question(self, question: str, session_id: Optional[str] = None, 
                        max_context_papers: int = 3) -> Dict:
        """Ask a question to the chat agent."""
        data = {
            "question": question,
            "max_context_papers": max_context_papers
        }
        if session_id:
            data["session_id"] = session_id
        return self._make_request("POST", "/chat/question", data=data)
    
    def clear_chat_session(self, session_id: str) -> Dict:
        """Clear a specific chat session history."""
        return self._make_request("DELETE", f"/chat/session/{session_id}")
