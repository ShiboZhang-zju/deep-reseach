"""Shared meaning of question-evidence relation types.

Coverage scoring and gap mining both answer the question "does this evidence
speak in favour of this research question?", and they used to answer it
differently: coverage counted only `supports` and filed `partially_answers` under
background, while gap mining accepted both. That split is not cosmetic, because
the matcher's own prompt defines `partially_answers` as "partially answers /
points out a limitation" and the heuristic fallback maps every `limitation` unit
to it — exactly the evidence gap mining depends on.

Measured consequence on a real run (task 5c2de9c7): the model returned 60
`background`, 21 `partially_answers` and zero `supports`, so all twelve research
questions scored 0.15 coverage and stayed `open`, while the same evidence was
admissible for mining. A different topic where the model happened to emit
`supports` scored 1.00 on comparable material. One definition, used by both.
"""

# Relations that count as corroboration for a research question.
SUPPORTING_RELATIONS = frozenset({"supports", "partially_answers"})

# Relations that count as evidence against the question's premise.
CONTRADICTING_RELATIONS = frozenset({"contradicts"})

# Below this relevance a match is too weak to count either way. Gap-mining
# admission has always applied this threshold; coverage now applies the same one,
# so a question cannot look corroborated by links that mining would reject.
MIN_RELATION_RELEVANCE = 0.5
