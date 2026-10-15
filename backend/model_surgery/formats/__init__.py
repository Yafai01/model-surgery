"""Readers for model file formats. Every reader treats its input as untrusted."""


class FormatError(ValueError):
    """Raised when a file is malformed, truncated, or violates a safety limit."""


class UnsupportedFormatError(ValueError):
    """Raised when a file is recognised but deliberately not parsed (e.g. pickle)."""
