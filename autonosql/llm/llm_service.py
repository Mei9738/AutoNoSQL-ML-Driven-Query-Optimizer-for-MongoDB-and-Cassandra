"""LLM service for query optimization suggestions"""

import os
import json
from typing import Dict, List, Optional
from abc import ABC, abstractmethod


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers"""

    @abstractmethod
    def generate_optimization(self, query: str, analysis: Dict) -> str:
        """Generate optimization suggestions using LLM"""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider"""

    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

    def generate_optimization(self, query: str, analysis: Dict) -> str:
        """Generate optimization suggestions using OpenAI"""
        prompt = self._build_prompt(query, analysis)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a database optimization expert specializing in NoSQL databases (MongoDB and Cassandra). Provide clear, actionable optimization suggestions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=800
            )

            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating optimization: {str(e)}"

    def _build_prompt(self, query: str, analysis: Dict) -> str:
        """Build the prompt for the LLM"""
        prompt = f"""Analyze this {analysis['database_type']} query and provide optimization recommendations:

Query:
{query}

Current Analysis:
- Issues Found: {', '.join(analysis['issues_found']) if analysis['issues_found'] else 'None'}
- Initial Suggestions: {', '.join(analysis['suggestions']) if analysis['suggestions'] else 'None'}
- Performance Metrics: {json.dumps(analysis['performance_metrics'], indent=2)}

Please provide:
1. A brief explanation of the main performance issues (if any)
2. Specific, actionable optimization recommendations
3. Example of an optimized version of the query (if applicable)
4. Best practices for this type of query

Keep your response concise and practical."""
        return prompt


class OllamaProvider(BaseLLMProvider):
    """Ollama local LLM provider"""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2"):
        self.base_url = base_url.rstrip('/')
        self.model = model
        try:
            import requests
            self.requests = requests
        except ImportError:
            raise ImportError("requests package not installed. Run: pip install requests")

    def generate_optimization(self, query: str, analysis: Dict) -> str:
        """Generate optimization suggestions using Ollama"""
        prompt = self._build_prompt(query, analysis)

        try:
            response = self.requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 500
                    }
                },
                timeout=180
            )

            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'No response generated')
            else:
                return f"Error: Ollama returned status {response.status_code}"

        except self.requests.exceptions.ConnectionError:
            return "Error: Could not connect to Ollama. Make sure Ollama is running (ollama serve)"
        except Exception as e:
            return f"Error generating optimization: {str(e)}"

    def _build_prompt(self, query: str, analysis: Dict) -> str:
        """Build the prompt for the LLM"""
        prompt = f"""You are a database optimization expert. Analyze this {analysis['database_type']} query and provide optimization recommendations.

Query:
{query}

Current Analysis:
- Issues Found: {', '.join(analysis['issues_found']) if analysis['issues_found'] else 'None'}
- Initial Suggestions: {', '.join(analysis['suggestions']) if analysis['suggestions'] else 'None'}
- Performance Metrics: {json.dumps(analysis['performance_metrics'], indent=2)}

Provide:
1. Brief explanation of main performance issues
2. Specific optimization recommendations
3. Example of optimized query if applicable
4. Best practices

Keep response concise and practical."""
        return prompt


class LLMService:
    """Service for interacting with LLMs for query optimization"""

    def __init__(self, provider: str = "ollama", **kwargs):
        """
        Initialize LLM service

        Args:
            provider: 'openai' or 'ollama'
            **kwargs: Provider-specific arguments
                For OpenAI: api_key, model (default: gpt-3.5-turbo)
                For Ollama: base_url (default: http://localhost:11434), model (default: llama3.2)
        """
        self.provider_name = provider.lower()

        if self.provider_name == "openai":
            api_key = kwargs.get('api_key') or os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable or pass api_key parameter")
            model = kwargs.get('model', 'gpt-3.5-turbo')
            self.provider = OpenAIProvider(api_key, model)

        elif self.provider_name == "ollama":
            base_url = kwargs.get('base_url') or os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
            model = kwargs.get('model') or os.getenv('OLLAMA_MODEL', 'llama3.2')
            self.provider = OllamaProvider(base_url, model)

        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'openai' or 'ollama'")

    def get_optimization_suggestions(self, query: str, analysis_result) -> str:
        """
        Get LLM-powered optimization suggestions

        Args:
            query: The database query
            analysis_result: QueryAnalysis object from analyzer

        Returns:
            Optimization suggestions as string
        """
        analysis_dict = {
            'database_type': analysis_result.database_type,
            'issues_found': analysis_result.issues_found,
            'suggestions': analysis_result.suggestions,
            'performance_metrics': analysis_result.performance_metrics
        }

        return self.provider.generate_optimization(query, analysis_dict)
