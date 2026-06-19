"""Multi-Modal Evidence Review — damage-claim evidence verification package.

A claim is reviewed in two stages:

1. Perception  — one multimodal Claude call per claim returns structured raw
   observations (per-image quality/authenticity, visible issue/part/severity,
   instruction detection) via forced tool use. See ``llm_client`` + ``prompts``.
2. Adjudication — pure-Python deterministic rules map perception + user history +
   evidence requirements onto the 14 output columns. See ``adjudicator``.

This split keeps the model focused on *seeing* and Python on *policy*, which makes
the system reproducible (temperature-free + disk cache) and cheap to iterate.
"""

__version__ = "1.0.0"
