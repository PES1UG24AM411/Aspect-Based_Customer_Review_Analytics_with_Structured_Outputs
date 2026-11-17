"""Bounded retry policy for semantic and transient failures."""
MAX_SEMANTIC_ATTEMPTS = 3
MAX_TRANSIENT_ATTEMPTS = 3
TRANSIENT_HTTP_STATUSES = (429, 500, 502, 503, 504)

class ExtractionFailure(RuntimeError):
    def __init__(self, category, message, attempts):
        super().__init__(message)
        self.category, self.attempts = category, attempts
