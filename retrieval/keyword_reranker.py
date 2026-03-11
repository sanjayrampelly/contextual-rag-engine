import re
from collections import Counter

class KeywordRanker:
    def tokenize(self, text: str) -> list[str]:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return text.split()

    def score(self, query: str, chunk: str) -> float:
        q_tokens = self.tokenize(query)
        c_tokens = self.tokenize(chunk)

        if not q_tokens or not c_tokens:
            return 0.0

        chunk_counts = Counter(c_tokens)

        overlap = 0
        for t in q_tokens:
            overlap += chunk_counts.get(t, 0)

        return overlap / len(q_tokens)

    def rerank(self, query: str, matches: list[dict]) -> list[dict]:
        for m in matches:
            text = m["metadata"]["text"]
            m["keyword_score"] = self.score(query, text)

        return sorted(
            matches,
            key=lambda x: (x["keyword_score"], x["score"]),
            reverse=True
        )
