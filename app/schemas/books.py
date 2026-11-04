from pydantic import BaseModel


class BookPassage(BaseModel):
    """One book passage that search_books retrieved for a chat reply."""
    chunk_id   : str
    book       : str | None = None
    author     : str | None = None
    section    : str
    pages      : str
    citation   : str
    similarity : float | None = None          # cosine to the question, 0–1; None when meaning search was down
    matched_by : list[str] = []
    text       : str                          # clipped to agent_tools.MAX_BOOK_TEXT
    weak       : bool = False                 # its search had no_book_support
