"""
Perplexity API integration for web search functionality
Provides real-time internet search when information isn't available in documents or internal knowledge
"""

import requests
import json
import os
from typing import Tuple, List
import logging

class PerplexitySearch:
    def __init__(self):
        """Initialize Perplexity search with API key."""
        self.api_key = os.getenv('PERPLEXITY_API_KEY')
        self.base_url = "https://api.perplexity.ai/chat/completions"
        # Current Perplexity model names (2024/2025)
        self.model = "sonar"
        
    def is_available(self) -> bool:
        """Check if Perplexity API is available."""
        return self.api_key is not None and self.api_key.strip() != ""
    
    def search_web(self, query: str, max_tokens: int = 500) -> Tuple[str, List[str]]:
        """
        Search the web using Perplexity API for real-time information.
        
        Args:
            query: The search query
            max_tokens: Maximum tokens for response
            
        Returns:
            Tuple of (answer, sources)
        """
        if not self.is_available():
            return "Web search is not available - API key not configured.", []
        
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "Be precise and concise. Provide accurate, up-to-date information with clear explanations."
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": 0.2,
                "top_p": 0.9,
                "return_images": False,
                "return_related_questions": False,
                "search_recency_filter": "month",
                "stream": False,
                "presence_penalty": 0,
                "frequency_penalty": 1
            }
            
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract the answer
                answer = data['choices'][0]['message']['content']
                
                # Extract citations/sources
                sources = data.get('citations', [])
                
                return answer, sources
            else:
                logging.error(f"Perplexity API error: {response.status_code} - {response.text}")
                return f"Web search failed with status {response.status_code}", []
                
        except requests.exceptions.Timeout:
            return "Web search timed out. Please try again.", []
        except requests.exceptions.RequestException as e:
            logging.error(f"Network error during web search: {str(e)}")
            return "Network error during web search. Please check your connection.", []
        except Exception as e:
            logging.error(f"Unexpected error during web search: {str(e)}")
            return f"Web search error: {str(e)}", []
    
    def enhanced_search(self, user_query: str, context_from_docs: str = None) -> Tuple[str, List[str]]:
        """
        Enhanced search that determines if web search is needed and formats the response.
        
        Args:
            user_query: The user's question
            context_from_docs: Context from uploaded documents
            
        Returns:
            Tuple of (enhanced_answer, web_sources)
        """
        if not self.is_available():
            return "Web search functionality requires a Perplexity API key.", []
        
        # Determine if web search is needed
        search_indicators = [
            "latest", "recent", "current", "news", "today", "now", "update",
            "what happened", "breaking", "2024", "2025", "stock price",
            "weather", "score", "result", "when did", "who won"
        ]
        
        needs_web_search = any(indicator in user_query.lower() for indicator in search_indicators)
        
        # Always try web search for real-time information or when no context
        if needs_web_search or not context_from_docs:
            # Create a focused search query
            search_query = user_query  # Use the query directly for better results
            web_answer, sources = self.search_web(search_query)
            
            if web_answer and "Web search" not in web_answer and "failed" not in web_answer.lower():
                # Combine with document context if available
                if context_from_docs and len(context_from_docs) > 50:
                    combined_answer = f"**Latest Information from Web:**\n{web_answer}\n\n**From Your Documents:**\n{context_from_docs}"
                else:
                    combined_answer = f"**Latest Information:**\n{web_answer}"
                
                return combined_answer, sources
        
        # Also try web search for general questions when no good context
        if not context_from_docs or len(context_from_docs) < 50:
            web_answer, sources = self.search_web(user_query)
            if web_answer and "Web search" not in web_answer and "failed" not in web_answer.lower():
                return f"**Web Search Results:**\n{web_answer}", sources
        
        return "", []
    
    def test_connection(self) -> Tuple[bool, str]:
        """Test the Perplexity API connection."""
        if not self.is_available():
            return False, "API key not configured"
        
        try:
            answer, sources = self.search_web("What is the capital of France?", max_tokens=50)
            if "Paris" in answer:
                return True, f"Connection successful. Found {len(sources)} sources."
            else:
                return False, f"Unexpected response: {answer}"
        except Exception as e:
            return False, f"Connection failed: {str(e)}"