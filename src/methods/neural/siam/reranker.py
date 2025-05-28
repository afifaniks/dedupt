import json
from typing import List, Tuple

import openai


def rerank_with_llm(
    query_stack: str,
    candidates: List[Tuple[int, float, str]],  # (id, vector_score, content)
    model: str = "gpt-4o",
) -> List[Tuple[int, float]]:
    """
    Rerank stack trace candidates using an LLM.
    
    Args:
        query_stack: The query stack trace.
        candidates: List of tuples (id, vector_score, content).
        model: OpenAI model name.
        
    Returns:
        List of tuples (id, llm_score) sorted by llm_score descending.
    """

    prompt = (
        "You are a software debugging assistant. Given a query stack trace and multiple candidate stack traces, "
        "rank the candidates by how similar they are to the query. Focus on error types, method/class similarity, "
        "call structure, and relevant messages.\n\n"
        f"Query stack trace:\n{query_stack}\n\n"
        "Candidates:\n"
    )

    for idx, (cid, _, content) in enumerate(candidates):
        prompt += f"[{cid}]:\n{content}\n\n"

    prompt += (
        "Respond with a JSON list of tuples where each tuple is (id, similarity_score), "
        "with a score from 0.0 (not similar) to 1.0 (very similar). Example: [[123, 0.91], [456, 0.72]]"
    )

    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert in analyzing software stack traces."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    try:
        llm_output = response["choices"][0]["message"]["content"]
        reranked = json.loads(llm_output)
        return sorted(reranked, key=lambda x: x[1], reverse=True)
    except Exception as e:
        print("Failed to parse LLM output:", e)
        return [(cid, vs) for cid, vs, _ in candidates]  # fallback to original order
