class FootballApiError(RuntimeError):
    """Base class for API-Sports domain failures."""


class FootballApiQuotaError(FootballApiError):
    """Raised when API-Sports returns rate-limit or quota exhaustion signals."""


class FootballApiTransientError(FootballApiError):
    """Raised when a retryable API-Sports failure exhausts retries."""


class FootballApiPayloadError(FootballApiError):
    """Raised when API-Sports returns a non-quota provider error envelope."""

