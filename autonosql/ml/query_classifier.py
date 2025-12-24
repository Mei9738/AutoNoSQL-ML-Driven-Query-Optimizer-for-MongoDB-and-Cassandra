"""
ML-based Query Classifier
Predicts whether a query needs optimization (good/bad classification)
"""

import os
import json
import pickle
import numpy as np
from typing import Dict, List, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib


class QueryFeatureExtractor:
    """Extract features from database queries for ML model"""

    @staticmethod
    def extract_mongodb_features(query_dict: dict) -> Dict[str, float]:
        """
        Extract features from MongoDB query

        Features:
        - has_regex: Using regex patterns
        - has_ne_nin: Using $ne or $nin operators
        - has_or: Using $or operator
        - filter_field_count: Number of fields in filter
        - has_sort: Has sorting
        - has_limit: Has limit
        - has_projection: Has field projection
        - uses_text_search: Using $text operator
        """
        features = {}
        filter_query = query_dict.get('filter', {})

        # Check for regex
        features['has_regex'] = float(QueryFeatureExtractor._has_operator(filter_query, '$regex'))

        # Check for $ne or $nin
        features['has_ne_nin'] = float(
            QueryFeatureExtractor._has_operator(filter_query, '$ne') or
            QueryFeatureExtractor._has_operator(filter_query, '$nin')
        )

        # Check for $or
        features['has_or'] = float(QueryFeatureExtractor._has_operator(filter_query, '$or'))

        # Count fields in filter
        features['filter_field_count'] = float(len(filter_query))

        # Check for sort
        features['has_sort'] = float('sort' in query_dict)

        # Check for limit
        features['has_limit'] = float('limit' in query_dict)

        # Check for projection
        features['has_projection'] = float('projection' in query_dict or 'fields' in query_dict)

        # Check for text search
        features['uses_text_search'] = float(QueryFeatureExtractor._has_operator(filter_query, '$text'))

        # Check for select all pattern
        features['no_filter'] = float(len(filter_query) == 0)

        return features

    @staticmethod
    def extract_cassandra_features(query_str: str) -> Dict[str, float]:
        """
        Extract features from Cassandra CQL query

        Features:
        - has_allow_filtering: Using ALLOW FILTERING
        - select_all: Using SELECT *
        - has_where: Has WHERE clause
        - has_in_clause: Using IN operator
        - in_clause_size: Size of IN clause (if present)
        - has_limit: Has LIMIT clause
        - has_order_by: Has ORDER BY clause
        """
        features = {}
        query_lower = query_str.lower()

        # Check for ALLOW FILTERING
        features['has_allow_filtering'] = float('allow filtering' in query_lower)

        # Check for SELECT *
        features['select_all'] = float('select *' in query_lower or 'select*' in query_lower)

        # Check for WHERE clause
        features['has_where'] = float('where' in query_lower)

        # Check for IN clause
        import re
        in_match = re.search(r'in\s*\([^)]+\)', query_lower)
        features['has_in_clause'] = float(bool(in_match))

        # Count IN clause size
        if in_match:
            in_values = in_match.group()
            features['in_clause_size'] = float(in_values.count(',') + 1)
        else:
            features['in_clause_size'] = 0.0

        # Check for LIMIT
        features['has_limit'] = float('limit' in query_lower)

        # Check for ORDER BY
        features['has_order_by'] = float('order by' in query_lower)

        # Check for no WHERE clause (full table scan)
        features['no_where'] = float('where' not in query_lower and 'select' in query_lower)

        return features

    @staticmethod
    def _has_operator(query_dict: dict, operator: str) -> bool:
        """Recursively check if an operator exists in query dict"""
        if not isinstance(query_dict, dict):
            return False

        for key, value in query_dict.items():
            if key == operator:
                return True
            if isinstance(value, dict):
                if QueryFeatureExtractor._has_operator(value, operator):
                    return True
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and QueryFeatureExtractor._has_operator(item, operator):
                        return True
        return False


class QueryClassifier:
    """ML-based query classifier to predict if a query needs optimization"""

    def __init__(self, model_path: str = None):
        """
        Initialize the classifier

        Args:
            model_path: Path to saved model file (optional)
        """
        self.model = None
        self.feature_names = None
        self.model_path = model_path or 'models/query_classifier.pkl'

        # Try to load existing model
        if os.path.exists(self.model_path):
            self.load_model()
        else:
            # Initialize with default Random Forest
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )

    def train(self, queries: List[Dict], labels: List[int], database_type: str = 'mongodb'):
        """
        Train the classifier

        Args:
            queries: List of query dicts (MongoDB) or query strings (Cassandra)
            labels: List of labels (0 = good, 1 = needs optimization)
            database_type: 'mongodb' or 'cassandra'
        """
        # Extract features
        features_list = []
        for query in queries:
            if database_type == 'mongodb':
                features = QueryFeatureExtractor.extract_mongodb_features(query)
            else:
                features = QueryFeatureExtractor.extract_cassandra_features(query)
            features_list.append(features)

        # Convert to numpy array
        self.feature_names = list(features_list[0].keys())
        X = np.array([[f[name] for name in self.feature_names] for f in features_list])
        y = np.array(labels)

        # Train model
        print(f"Training classifier with {len(X)} samples...")
        self.model.fit(X, y)

        # Print feature importances
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            print("\nFeature Importances:")
            for name, importance in sorted(zip(self.feature_names, importances),
                                          key=lambda x: x[1], reverse=True):
                print(f"  {name:25s}: {importance:.4f}")

        return self

    def predict(self, query, database_type: str = 'mongodb') -> Tuple[int, float]:
        """
        Predict if query needs optimization

        Args:
            query: Query dict (MongoDB) or query string (Cassandra)
            database_type: 'mongodb' or 'cassandra'

        Returns:
            (prediction, confidence) where prediction is 0 (good) or 1 (bad)
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")

        # Extract features
        if database_type == 'mongodb':
            features = QueryFeatureExtractor.extract_mongodb_features(query)
        else:
            features = QueryFeatureExtractor.extract_cassandra_features(query)

        # Convert to array
        X = np.array([[features[name] for name in self.feature_names]])

        # Predict
        prediction = self.model.predict(X)[0]

        # Get confidence (probability)
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)[0]
            confidence = probabilities[prediction]
        else:
            confidence = 1.0

        return int(prediction), float(confidence)

    def evaluate(self, queries: List[Dict], labels: List[int], database_type: str = 'mongodb'):
        """Evaluate model on test data"""
        # Extract features
        features_list = []
        for query in queries:
            if database_type == 'mongodb':
                features = QueryFeatureExtractor.extract_mongodb_features(query)
            else:
                features = QueryFeatureExtractor.extract_cassandra_features(query)
            features_list.append(features)

        X = np.array([[f[name] for name in self.feature_names] for f in features_list])
        y = np.array(labels)

        predictions = self.model.predict(X)
        accuracy = accuracy_score(y, predictions)

        print(f"\nModel Accuracy: {accuracy:.2%}")
        print("\nClassification Report:")
        print(classification_report(y, predictions, target_names=['Good Query', 'Needs Optimization']))

        return accuracy

    def save_model(self, path: str = None):
        """Save the trained model"""
        path = path or self.model_path
        os.makedirs(os.path.dirname(path), exist_ok=True)

        model_data = {
            'model': self.model,
            'feature_names': self.feature_names
        }

        joblib.dump(model_data, path)
        print(f"Model saved to {path}")

    def load_model(self, path: str = None):
        """Load a trained model"""
        path = path or self.model_path

        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        model_data = joblib.load(path)
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']

        print(f"Model loaded from {path}")
