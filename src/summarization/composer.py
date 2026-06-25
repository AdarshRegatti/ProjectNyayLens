# src/summarization/composer.py
def compose(sentences, scores, top_k=5):
    """
    Select top-k sentences by Legal-BERT score, then RESTORE their original
    document order before returning. This gives PEGASUS a coherent narrative
    instead of a randomly ordered bag of sentences.
    """
    # Tag each sentence with its original index
    indexed = list(enumerate(zip(sentences, scores)))
    
    # Pick top-k by score
    top = sorted(indexed, key=lambda x: x[1][1], reverse=True)[:top_k]
    
    # Re-sort by original document position for narrative coherence
    top_in_order = sorted(top, key=lambda x: x[0])
    
    return [s for _, (s, _) in top_in_order]
