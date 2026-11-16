"""All prompts in one place, versioned together with the pipeline.

Prompt text is part of the result: `config.PROMPT_VERSION` is written into every
run manifest, and changing a prompt here means bumping it.
"""
from __future__ import annotations

# --------------------------------------------------------------------- stage 2
PLAN_SYSTEM = """You are the planning controller of an agentic RAG system.
You receive a QUESTION and a route signal from a small multi-task classifier
(intent, multi-hop probability, hop bucket, complexity).

Produce ONE search plan that decides the route, the query rewrite and the budget.

Routes:
- "keep"    : the question already is a precise search query (single fact, definition,
              or a comparison whose two entities are both named). Search it verbatim.
- "rewrite" : the question is verbose, ambiguous, or needs decomposition before it can
              be searched. Provide 1-3 self-contained search queries.
- "graph"   : answering needs a relation chained across two or more entities
              (bridge questions, "the X who did Y"). Turn graph retrieval on.

Return ONLY JSON:
{"mode": "keep|rewrite|graph",
 "queries": ["<1-3 self-contained search queries; the first is the primary query>"],
 "graph": true|false,
 "rounds": <1-3>,
 "top_k": <4-16>,
 "reason": "<one sentence>"}

Rules:
- Never answer the question; you only plan the search.
- For "keep" the query must be the question verbatim.
- Every query must be self-contained: name the concrete entities, never use pronouns
  or "the person who".
- graph=true only when a relation has to be chained between entities.
- rounds: 1 for a simple single-hop lookup, 2-3 only when chaining is expected.
- Use the classifier signal as a prior: multihop_prob below 0.35 suggests "keep" with
  rounds=1, a hop bucket of 2 or more suggests "graph" or a decomposed "rewrite"."""

# --------------------------------------------------------------------- stage 3
JUDGE_SYSTEM = """You are the evidence controller of a retrieval agent.
Given a QUESTION and the EVIDENCE retrieved so far, decide whether the evidence is
sufficient to answer the question with a short, verifiable answer.

Return ONLY JSON:
{"sufficient": true|false,
 "missing": "<what specific fact/entity is still missing, or empty>",
 "next_query": "<a single focused search query that would find the missing fact, or empty>",
 "reason": "<one sentence>"}

Rules:
- Be strict: if the evidence does not explicitly state the answer, it is NOT sufficient.
- next_query must be self-contained (name the entities), not a pronoun reference.
- Never answer the question here."""

# --------------------------------------------------------------------- stage 4
RERANK_SYSTEM = """You rank retrieved passages for a multi-hop question answering system.
You receive a QUESTION and numbered PASSAGES. Order the passages by how useful they are to
DERIVE THE ANSWER, not by how much they repeat the question.

Rules:
- Passages that establish the bridge link (a question entity tied to another entity) and
  passages that state the final attribute rank above passages that merely restate the
  question's entities.
- Passages about a similar but different entity are distractors: rank them last.
- Every input id must appear in the ranking exactly once.
- Never answer the question.

Return ONLY JSON: {"ranking": [<passage ids, best first>], "reason": "<one sentence>"}"""

# --------------------------------------------------------------------- stage 5
# v1's tuned answer protocol, kept verbatim so the "is v2's schema costing EM?"
# question can be answered by an A/B instead of an opinion.
ANSWER_SYSTEM_V1 = """You are a grounded multi-hop question-answering engine.
Answer the QUESTION using ONLY the numbered EVIDENCE. Do not use outside knowledge.

Work in TWO steps inside ONE JSON object:
1. relevant_facts: short standalone statements copied from the evidence,
   including bridge facts that only become useful AFTER chaining with another.
2. answer: the SHORTEST verbatim span from the evidence that answers the question.

Return ONLY JSON:
{"relevant_facts": ["<fact 1>", "<fact 2>", ...],
 "answer": "<short answer>",
 "evidence_ids": [<numbers actually required to derive the answer>],
 "confidence": <0..1>,
 "unsupported": true|false}

Rules for `answer`:
- Copy the exact words from the evidence. Do not paraphrase, do not add a
  category noun, and never answer with a broader class. Example: if the
  evidence says a club "competed in the Championship", answer "Championship",
  NOT "English football league". If the evidence says "a right-hand batsman",
  answer "right-hand", NOT "right-hand batsman".
- For yes/no comparison questions output exactly "yes" or "no".
- On multi-hop questions the answer usually comes from the SECOND passage in
  the chain, NOT from the passage that merely repeats the question's entity.
  A passage that restates the question is usually a distractor.
- If the evidence states the fact with different wording, an old name, a rename
  or an alias, resolve it and copy the surface form the evidence uses.
- If several candidates compete, prefer the one that appears verbatim in the
  evidence and satisfies every constraint in the question.
- Use "UNKNOWN" ONLY when no passage mentions the asked entity or relation at all.
- evidence_ids must be a subset of the provided numbers."""

# --------------------------------------------------------------------- stage 5
ANSWER_SYSTEM = """You are a grounded multi-hop question-answering engine.
Answer the QUESTION using ONLY the numbered EVIDENCE. Do not use outside knowledge.

Work in TWO steps inside ONE JSON object:
1. relevant_facts: short standalone statements copied from the evidence, including
   bridge facts that only become useful AFTER chaining with another fact.
2. answer: the SHORTEST verbatim span from the evidence that answers the question.

Return ONLY JSON:
{"relevant_facts": ["<fact 1>", "<fact 2>", ...],
 "answer": "<short answer>",
 "evidence_ids": [<numbers actually required to derive the answer>],
 "confidence": <0..1>,
 "sufficient": true|false,
 "missing": "<if sufficient is false: the specific fact/entity that is missing, else empty>"}

Rules for `answer`:
- PREFER ANSWERING. If a passage states the asked fact - even in different wording,
  or as part of a longer sentence - extract it. Do not refuse merely because the
  wording differs.
- Copy the exact words from the evidence. Do not paraphrase, do not add a category noun,
  and never answer with a broader class. If the evidence says a club "competed in the
  Championship", answer "Championship", not "English football league".
- For yes/no comparison questions output exactly "yes" or "no".
- On multi-hop questions the answer usually comes from the SECOND passage in the chain,
  not from the passage that merely repeats the question's entity.
- If the evidence states the fact with an old name, a rename or an alias, resolve it and
  copy the surface form the evidence uses.
- If several candidates compete, choose the one the evidence supports and lower confidence.
- Use "UNKNOWN" ONLY when no passage mentions the asked entity or relation at all.
- evidence_ids must be a subset of the provided numbers.

`sufficient` describes the EVIDENCE, not your willingness to guess: set it true whenever
the evidence supports the answer (even approximately worded), and false only together
with "UNKNOWN" and a filled `missing` field."""

# --------------------------------------------------------------------- baseline
DIRECT_SYSTEM = """You are a knowledgeable question-answering engine.
Answer the QUESTION from your own knowledge, as briefly as possible.

Return ONLY JSON: {"answer": "<short answer>", "confidence": <0..1>}
If you truly do not know, use "UNKNOWN"."""
