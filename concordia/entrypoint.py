"""DEPRECATED: Use aurelius/simulation/entrypoint.py instead.

This file is kept for backward compatibility. The canonical simulation
entrypoint is at aurelius/simulation/entrypoint.py with both Concordia
library and OpenAI fallback paths.
"""
# Re-export for backward compatibility
from aurelius.simulation.entrypoint import run_simulation, main  # noqa: F401

if __name__ == "__main__":
    main()
