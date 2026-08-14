import os
os.environ["MOCK_LLM"] = "1"

# Mock the globals before importing app to avoid loading heavy models
import sys
import unittest.mock as mock

class MockEmbedder:
    def encode(self, text):
        import numpy as np; return np.array([0.0]*384)

class MockCollection:
    def add(self, *args, **kwargs):
        pass
    def query(self, *args, **kwargs):
        return {"documents": [["Zepto Return Policy: Perishable items cannot be returned once delivered."]], "distances": [[0.1]]}

class MockChromaClient:
    def create_collection(self, *args, **kwargs):
        return MockCollection()

sys.modules['sentence_transformers'] = mock.Mock()
sys.modules['sentence_transformers'].SentenceTransformer = lambda x: MockEmbedder()
sys.modules['chromadb'] = mock.Mock()
sys.modules['chromadb'].Client = lambda: MockChromaClient()

# Now import app
from app import workflow

# Test Policy Query
policy_state = {
    "query": "What is the return policy?",
    "intent": "",
    "retrieved_chunk": "",
    "answer": "",
    "sources": [],
    "confidence": 0.0
}
res_policy = workflow.invoke(policy_state)
print("POLICY TEST:", res_policy["answer"])

# Test General Query
general_state = {
    "query": "Hello there",
    "intent": "",
    "retrieved_chunk": "",
    "answer": "",
    "sources": [],
    "confidence": 0.0
}
res_general = workflow.invoke(general_state)
print("GENERAL TEST:", res_general["answer"])
