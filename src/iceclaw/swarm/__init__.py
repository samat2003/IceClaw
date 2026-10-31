"""Swarm runtime: append-only evidence store, event bus, budgets and termination.

The runtime never decides scientific content or ordering; it only validates references, enforces
budgets and detects quiescence. It must not import ``iceclaw.evaluation`` (tested).
"""
