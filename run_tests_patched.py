import sys
import unittest
from unittest.mock import patch


def fake_embed_documents(self, texts):
    return [[0.1] * 8 for _ in texts]


def fake_embed_query(self, text):
    return [0.1] * 8


patchers = [
    patch("langchain_ollama.OllamaEmbeddings.embed_documents", fake_embed_documents),
    patch("langchain_ollama.OllamaEmbeddings.embed_query", fake_embed_query),
]
for p in patchers:
    p.start()

loader = unittest.TestLoader()
suite = loader.discover("tests")
runner = unittest.TextTestRunner(verbosity=1)
result = runner.run(suite)

for p in patchers:
    p.stop()

sys.exit(0 if result.wasSuccessful() else 1)
