"""Global test configuration. Runs before any test module is imported."""

import os

# Set a valid JWT secret for all tests (P0-2 requires >= 32 chars, not a known weak value)
os.environ.setdefault("JWT_SECRET", "test-secret-key-that-is-at-least-32-characters-long")
