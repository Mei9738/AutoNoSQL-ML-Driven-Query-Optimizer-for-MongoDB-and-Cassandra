"""MongoDB query analyzer"""

import json
from typing import Dict, List, Any
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from ..base_analyzer import BaseQueryAnalyzer, QueryAnalysis


class MongoDBAnalyzer(BaseQueryAnalyzer):
    """Analyzer for MongoDB queries"""

    def __init__(self, connection_params: Dict[str, Any]):
        super().__init__(connection_params)
        self.database_name = connection_params.get('database', 'test')

    def connect(self) -> bool:
        """Establish connection to MongoDB"""
        try:
            uri = self.connection_params.get('uri', 'mongodb://localhost:27017')
            self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            # Test connection
            self.client.admin.command('ping')
            print(f"Connected to MongoDB at {uri}")
            return True
        except PyMongoError as e:
            print(f"Failed to connect to MongoDB: {e}")
            return False

    def disconnect(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            print("Disconnected from MongoDB")

    def analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze a MongoDB query"""
        issues = []
        suggestions = []
        performance_metrics = {}
        execution_plan = {}

        try:
            # Parse the query (assuming JSON format)
            if isinstance(query, str):
                # Try to parse as JSON
                try:
                    query_dict = json.loads(query)
                except json.JSONDecodeError:
                    # If it's a string representation of a dict, use eval (safe since we control input)
                    import ast
                    try:
                        query_dict = ast.literal_eval(query)
                    except:
                        raise ValueError("Query must be valid JSON or dict string")
            else:
                query_dict = query

            collection_name = query_dict.get('collection', 'unknown')
            operation = query_dict.get('operation', 'find')
            filter_query = query_dict.get('filter', {})

            # Get the collection
            db = self.client[self.database_name]
            collection = db[collection_name]

            # Get execution stats
            stats = self.get_execution_stats(query)
            performance_metrics = stats

            # Get execution plan
            if operation == 'find':
                explain_result = collection.find(filter_query).explain()
                execution_plan = explain_result

                # Analyze execution plan
                winning_plan = explain_result.get('queryPlanner', {}).get('winningPlan', {})
                stage = winning_plan.get('stage', '')

                if stage == 'COLLSCAN':
                    issues.append("Query uses COLLECTION SCAN (no index)")
                    suggestions.append("Consider adding an index on the queried fields")

                # Check for missing indexes
                index_suggestions = self.check_indexes(query)
                suggestions.extend(index_suggestions)

            # Check for inefficient patterns
            if filter_query:
                # Check for $regex without index
                for key, value in filter_query.items():
                    if isinstance(value, dict) and '$regex' in value:
                        issues.append(f"Using $regex on field '{key}' without index (slow)")
                        suggestions.append(f"Consider using text index on '{key}' or restructure query")

                    # Check for $ne or $nin
                    if isinstance(value, dict) and ('$ne' in value or '$nin' in value):
                        issues.append(f"Using $ne/$nin on field '{key}' (cannot use index efficiently)")
                        suggestions.append(f"Consider inverting logic or using $in with specific values")

            # Check collection stats
            coll_stats = db.command('collStats', collection_name)
            doc_count = coll_stats.get('count', 0)
            performance_metrics['document_count'] = doc_count

            if doc_count > 10000 and not collection.list_indexes():
                issues.append(f"Large collection ({doc_count} docs) with no indexes")
                suggestions.append("Add indexes for frequently queried fields")

        except json.JSONDecodeError:
            issues.append("Invalid JSON query format")
        except PyMongoError as e:
            issues.append(f"MongoDB error: {str(e)}")
        except Exception as e:
            issues.append(f"Analysis error: {str(e)}")

        return QueryAnalysis(
            query=query,
            database_type="MongoDB",
            issues_found=issues,
            suggestions=suggestions,
            performance_metrics=performance_metrics,
            execution_plan=execution_plan
        )

    def get_execution_stats(self, query: str) -> Dict[str, Any]:
        """Get MongoDB query execution statistics"""
        try:
            query_dict = json.loads(query) if isinstance(query, str) else query
            collection_name = query_dict.get('collection', 'unknown')
            filter_query = query_dict.get('filter', {})

            db = self.client[self.database_name]
            collection = db[collection_name]

            # Get execution stats
            explain_result = collection.find(filter_query).explain('executionStats')
            exec_stats = explain_result.get('executionStats', {})

            return {
                'execution_time_ms': exec_stats.get('executionTimeMillis', 0),
                'documents_examined': exec_stats.get('totalDocsExamined', 0),
                'documents_returned': exec_stats.get('nReturned', 0),
                'execution_success': exec_stats.get('executionSuccess', False)
            }
        except Exception as e:
            return {'error': str(e)}

    def check_indexes(self, query: str) -> List[str]:
        """Check if appropriate indexes exist for the query"""
        suggestions = []
        try:
            query_dict = json.loads(query) if isinstance(query, str) else query
            collection_name = query_dict.get('collection', 'unknown')
            filter_query = query_dict.get('filter', {})

            db = self.client[self.database_name]
            collection = db[collection_name]

            # Get existing indexes
            existing_indexes = list(collection.list_indexes())
            indexed_fields = set()
            for idx in existing_indexes:
                for field in idx.get('key', {}).keys():
                    indexed_fields.add(field)

            # Check if query fields are indexed
            if filter_query:
                query_fields = set(filter_query.keys())
                missing_indexes = query_fields - indexed_fields - {'_id'}

                for field in missing_indexes:
                    suggestions.append(f"Consider creating index on '{field}'")

        except Exception as e:
            suggestions.append(f"Error checking indexes: {str(e)}")

        return suggestions
