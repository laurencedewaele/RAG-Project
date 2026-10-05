class DomainError(Exception):
    """Base domain exception."""


class EmptyQuestionError(DomainError):
    """Raised when a user asks an empty question."""
