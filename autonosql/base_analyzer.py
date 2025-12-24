"""Base class for database query analyzers"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class QueryAnalysis:
    """Results from query analysis"""
    query: str
    database_type: str
    issues_found: List[str]
    suggestions: List[str]
    performance_metrics: Dict[str, Any]
    execution_plan: Dict[str, Any]


class BaseQueryAnalyzer(ABC):
    """Abstract base class for query analyzers"""

    def __init__(self, connection_params: Dict[str, Any]):
        self.connection_params = connection_params
        self.client = None

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the database"""
        pass

    @abstractmethod
    def disconnect(self):
        """Close database connection"""
        pass

    @abstractmethod
    def analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze a query and return findings"""
        pass

    @abstractmethod
    def get_execution_stats(self, query: str) -> Dict[str, Any]:
        """Get query execution statistics"""
        pass

    @abstractmethod
    def check_indexes(self, query: str) -> List[str]:
        """Check if appropriate indexes exist for the query"""
        pass

    def format_analysis(self, analysis: QueryAnalysis) -> str:
        """Format analysis results as a readable string"""
        output = []
        output.append(f"\n{'='*60}")
        output.append(f"Query Analysis - {analysis.database_type}")
        output.append(f"{'='*60}")
        output.append(f"\nQuery:\n{analysis.query}\n")

        if analysis.issues_found:
            output.append("Issues Found:")
            for i, issue in enumerate(analysis.issues_found, 1):
                output.append(f"  {i}. {issue}")
        else:
            output.append("No issues found!")

        if analysis.suggestions:
            output.append("\nSuggestions:")
            for i, suggestion in enumerate(analysis.suggestions, 1):
                output.append(f"  {i}. {suggestion}")

        if analysis.performance_metrics:
            output.append("\nPerformance Metrics:")
            for key, value in analysis.performance_metrics.items():
                output.append(f"  - {key}: {value}")

        output.append(f"{'='*60}\n")
        return "\n".join(output)
