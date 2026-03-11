class SimpleReranker:
    def rerank(self, matches):
        """
        Re-rank based on:
        1. Vector similarity score
        2. Chunk length (longer often = more informative)
        """
        reranked = sorted(
            matches,
            key=lambda m: (
                m["score"],
                len(m["metadata"]["text"])
            ),
            reverse=True
        )
        return reranked
