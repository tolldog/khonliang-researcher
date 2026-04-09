"""Researcher-flavored idea parser.

Thin subclass of khonliang_researcher.BaseIdeaParser that loads the
researcher's own prompt file. All decomposition logic lives in the lib —
this module just supplies the prompt.
"""

from khonliang_researcher import BaseIdeaParser

from researcher.roles import _load_prompt


class IdeaParserRole(BaseIdeaParser):
    """Decompose informal text into claims and search queries.

    Uses the researcher project's prompt file from prompts/idea_parser.md.
    """

    def __init__(self, model_pool, **kwargs):
        super().__init__(
            model_pool=model_pool,
            system_prompt=_load_prompt("idea_parser.md"),
            **kwargs,
        )
