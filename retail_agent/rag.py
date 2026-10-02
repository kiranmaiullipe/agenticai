"""Prototype policy RAG using lightweight TF-IDF retrieval.
Replace this retriever with the team's approved embedding/vector DB implementation later.
"""
from pathlib import Path
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

POLICY_DIR = Path(__file__).parent / "policies"

def _chunks(text, max_chars=700):
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out = []
    for p in paras:
        if len(p) <= max_chars:
            out.append(p)
        else:
            out.extend(p[i:i+max_chars] for i in range(0, len(p), max_chars))
    return out

def load_policy_chunks():
    docs = []
    for path in sorted(POLICY_DIR.glob("*.md")):
        for i, chunk in enumerate(_chunks(path.read_text(encoding="utf-8"))):
            docs.append({"source": path.name, "chunk": i, "text": chunk})
    return docs

class PolicyRetriever:
    def __init__(self):
        self.docs = load_policy_chunks()
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform([d["text"] for d in self.docs]) if self.docs else None

    def retrieve(self, query, k=3):
        if not self.docs or not query.strip():
            return []
        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.matrix).ravel()
        ids = scores.argsort()[::-1][:k]
        return [{**self.docs[i], "score": float(scores[i])} for i in ids if scores[i] > 0]

_retriever = None
def retrieve_policy(query, k=3):
    global _retriever
    if _retriever is None:
        _retriever = PolicyRetriever()
    return _retriever.retrieve(query, k)
