"""
bench/prompts.py — Fixed prompt set shared across all engines.

Keeping prompts fixed is critical for a fair benchmark: every engine sees
exactly the same input tokens, so timing differences come from the runtime,
not the input.

The prompt_id is the stable key used in results JSON.
"""

PROMPTS: list[dict] = [
    {
        "prompt_id": "short_factual",
        "system": "You are a helpful assistant. Be concise.",
        "user": "What is Apple Silicon and why does unified memory matter for AI workloads?",
        "max_tokens": 200,
    },
    {
        "prompt_id": "medium_reasoning",
        "system": "You are a senior SRE at a big data company.",
        "user": (
            "A Spark job on a 100-node cluster is taking 3x longer than its SLA. "
            "Walk me through your first 5 diagnostic steps."
        ),
        "max_tokens": 400,
    },
    {
        "prompt_id": "long_generation",
        "system": "You are a technical writer.",
        "user": (
            "Write a detailed runbook for responding to a Kafka consumer lag alert. "
            "Include: symptoms, root causes, remediation steps, and escalation criteria."
        ),
        "max_tokens": 600,
    },
]
