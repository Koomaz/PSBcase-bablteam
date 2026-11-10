"""ML2: generation, prompting and guardrails layer for the PSB Legal Assistant.

ML1 (`retrieval`) отвечает за поиск и отдаёт `RetrievedChunk`.
Этот пакет превращает их в доказательства, собирает промпт, вызывает LLM,
проверяет ответ и пишет submission.csv.
"""

from .evidence import build_evidence, parse_question, premise_warnings
from .prompt_builder import build_messages, load_system_prompt
from .validator import validate
from .csvio import read_questions, write_submission

__all__ = [
    "build_evidence",
    "parse_question",
    "premise_warnings",
    "build_messages",
    "load_system_prompt",
    "validate",
    "read_questions",
    "write_submission",
]
