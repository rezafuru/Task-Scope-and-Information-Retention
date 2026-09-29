"""Taskonomy task-family codec: dataset, models, edge target, and the training loop.

The entropy models come from CompressAI, which is an optional dependency. Importing this
package does not require it. Building or loading a codec does.
"""

__all__ = ["codec", "data", "edges", "models"]
