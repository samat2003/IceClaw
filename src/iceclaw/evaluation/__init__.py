"""Evaluator-owned code: sealed fault keys, expert-reference targets and scoring.

Import rule (tested): nothing under ``iceclaw.agents``, ``iceclaw.swarm``, ``iceclaw.science`` or
``iceclaw.data`` may import this package, and this package must not import agents/swarm.
Agents receive only ``AgentScenario`` objects (opaque case ids, public arrays, public context).
"""
