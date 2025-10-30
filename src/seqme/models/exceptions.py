class OptionalDependencyError(Exception):
    """Raised when an optional dependency is not installed."""

    def __init__(self, package: str):
        message = f'Optional dependency is missing. Install it with: pip install "seqme[{package}]".'
        super().__init__(message)
