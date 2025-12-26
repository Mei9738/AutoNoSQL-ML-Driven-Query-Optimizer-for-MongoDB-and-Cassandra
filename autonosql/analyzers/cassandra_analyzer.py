"""Cassandra query analyzer"""

import re
from typing import Dict, List, Any
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from ..base_analyzer import BaseQueryAnalyzer, QueryAnalysis

class CassandraAnalyzer(BaseQueryAnalyzer):
    """Analyzer for Cassandra CQL queries"""

    def __init__(self, connection_params: Dict[str, Any]):
        super().__init__(connection_params)
        self.keyspace = connection_params.get('keyspace', 'system')
        self.session = None

    def connect(self) -> bool:
        """Establish connection to Cassandra"""
        try:
            hosts = self.connection_params.get('hosts', ['127.0.0.1'])
            port = self.connection_params.get('port', 9042)

            # Handle authentication if provided
            auth_provider = None
            if 'username' in self.connection_params and 'password' in self.connection_params:
                auth_provider = PlainTextAuthProvider(
                    username=self.connection_params['username'],
                    password=self.connection_params['password']
                )

            self.client = Cluster(hosts, port=port, auth_provider=auth_provider)
            self.session = self.client.connect()

            # Set keyspace if provided
            if self.keyspace and self.keyspace != 'system':
                self.session.set_keyspace(self.keyspace)

            print(f"Connected to Cassandra at {hosts}:{port}")
            return True
        except Exception as e:
            print(f"Failed to connect to Cassandra: {e}")
            return False

    def disconnect(self):
        """Close Cassandra connection"""
        if self.client:
            self.client.shutdown()
            print("Disconnected from Cassandra")

    def analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze a Cassandra CQL query"""
        issues = []
        suggestions = []
        performance_metrics = {}
        execution_plan = {}

        try:
            # Clean up the query
            query = query.strip().rstrip(';')

            # Get execution stats
            stats = self.get_execution_stats(query)
            performance_metrics = stats

            # Parse table name from query
            table_name = self._extract_table_name(query)

            # Analyze query patterns
            query_lower = query.lower()

            # Check for ALLOW FILTERING
            if 'allow filtering' in query_lower:
                issues.append("Query uses ALLOW FILTERING (full table scan)")
                suggestions.append("Add appropriate indexes or redesign query to avoid ALLOW FILTERING")

            # Check for SELECT * on large tables
            if 'select *' in query_lower or 'select*' in query_lower:
                issues.append("Using SELECT * (may retrieve unnecessary data)")
                suggestions.append("Select only needed columns to reduce network overhead")

            # Check for missing WHERE clause
            if 'where' not in query_lower and 'select' in query_lower:
                issues.append("Query missing WHERE clause (will scan entire table)")
                suggestions.append("Add WHERE clause to filter on partition key")

            # Check for IN clause with many values
            in_match = re.search(r'in\s*\([^)]+\)', query_lower)
            if in_match:
                in_values = in_match.group()
                value_count = in_values.count(',') + 1
                if value_count > 10:
                    issues.append(f"IN clause with {value_count} values (may cause performance issues)")
                    suggestions.append("Consider breaking into multiple queries or redesigning data model")

            # Check for non-partition key WHERE clause
            if table_name:
                index_suggestions = self.check_indexes(query)
                suggestions.extend(index_suggestions)

            # Analyze table if we can identify it
            if table_name and self.keyspace:
                try:
                    # Get table metadata
                    metadata = self.client.metadata.keyspaces.get(self.keyspace)
                    if metadata:
                        table_meta = metadata.tables.get(table_name)
                        if table_meta:
                            execution_plan['partition_keys'] = [col.name for col in table_meta.partition_key]
                            execution_plan['clustering_keys'] = [col.name for col in table_meta.clustering_key]
                            execution_plan['columns'] = list(table_meta.columns.keys())

                            # Check if querying by partition key
                            partition_keys = {col.name for col in table_meta.partition_key}
                            where_fields = self._extract_where_fields(query)

                            if where_fields and not where_fields.intersection(partition_keys):
                                issues.append("Query does not filter by partition key")
                                suggestions.append(f"Consider filtering by partition key: {', '.join(partition_keys)}")

                except Exception as e:
                    suggestions.append(f"Could not analyze table metadata: {str(e)}")

        except Exception as e:
            issues.append(f"Analysis error: {str(e)}")

        return QueryAnalysis(
            query=query,
            database_type="Cassandra",
            issues_found=issues,
            suggestions=suggestions,
            performance_metrics=performance_metrics,
            execution_plan=execution_plan
        )

    def get_execution_stats(self, query: str) -> Dict[str, Any]:
        """Get Cassandra query execution statistics"""
        # Try to execute query with tracing if session exists
        if not self.session:
            return {
                'analysis_type': 'static',
                'note': 'No database connection - static analysis only'
            }

        try:
            # Execute query with tracing enabled to get performance metrics
            import time
            start_time = time.time()

            # Execute with trace=True to get detailed execution info
            future = self.session.execute_async(query, trace=True)
            result = future.result()

            execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            # Get trace information if available
            trace = future.get_query_trace(max_wait=2.0)

            stats = {
                'analysis_type': 'live',
                'execution_time_ms': round(execution_time, 2),
                'rows_returned': len(list(result)),
                'coordinator': str(trace.coordinator) if trace else None,
                'duration_micros': trace.duration.total_seconds() * 1000000 if trace else None
            }

            # Add trace events if available (useful for debugging performance)
            if trace and trace.events:
                stats['trace_events_count'] = len(trace.events)
                # Get the slowest event
                slowest = max(trace.events, key=lambda e: e.source_elapsed.total_seconds() if e.source_elapsed else 0)
                stats['slowest_operation'] = slowest.description if slowest else None

            return stats

        except Exception as e:
            # If execution fails (table doesn't exist, etc.), fall back to static analysis
            error_msg = str(e)

            # Check if it's a "table doesn't exist" error
            if 'unconfigured table' in error_msg.lower() or 'does not exist' in error_msg.lower():
                return {
                    'analysis_type': 'static',
                    'note': 'Table does not exist - static analysis only',
                    'suggestion': 'Create the table to enable live query execution'
                }

            return {
                'analysis_type': 'static',
                'note': 'Query execution failed - static analysis only',
                'error': error_msg[:100]  # Truncate long errors
            }

    def check_indexes(self, query: str) -> List[str]:
        """Check if appropriate indexes exist for the query"""
        suggestions = []
        try:
            table_name = self._extract_table_name(query)
            if not table_name or not self.keyspace:
                return suggestions

            # Get existing indexes
            index_query = f"""
                SELECT index_name, options
                FROM system_schema.indexes
                WHERE keyspace_name = '{self.keyspace}'
                AND table_name = '{table_name}'
            """

            rows = self.session.execute(index_query)
            existing_indexes = list(rows)

            # Extract WHERE clause fields
            where_fields = self._extract_where_fields(query)

            if where_fields and not existing_indexes:
                for field in where_fields:
                    suggestions.append(f"Consider creating secondary index on '{field}'")

        except Exception as e:
            suggestions.append(f"Error checking indexes: {str(e)}")

        return suggestions

    def _extract_table_name(self, query: str) -> str:
        """Extract table name from CQL query"""
        query_lower = query.lower()

        # FROM clause
        from_match = re.search(r'from\s+(\w+)', query_lower)
        if from_match:
            return from_match.group(1)

        # INSERT INTO
        insert_match = re.search(r'insert\s+into\s+(\w+)', query_lower)
        if insert_match:
            return insert_match.group(1)

        # UPDATE
        update_match = re.search(r'update\s+(\w+)', query_lower)
        if update_match:
            return update_match.group(1)

        return None

    def _extract_where_fields(self, query: str) -> set:
        """Extract field names from WHERE clause"""
        fields = set()
        query_lower = query.lower()

        where_match = re.search(r'where\s+(.+?)(?:allow filtering|order by|limit|$)', query_lower)
        if where_match:
            where_clause = where_match.group(1)
            # Find field names before comparison operators
            field_matches = re.findall(r'(\w+)\s*(?:=|>|<|>=|<=|in)', where_clause)
            fields.update(field_matches)

        return fields
