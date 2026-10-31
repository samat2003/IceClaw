import pytest


@pytest.fixture(autouse=True)
def _never_read_dotenv(monkeypatch):
    """Tests must never open the user's .env or depend on ambient secrets."""
    monkeypatch.setenv("ICECLAW_SKIP_DOTENV", "1")
    # P0-P3 tests assume prediction-only catalogs; P4 tests opt in via Settings(sombench_root=...)
    monkeypatch.setenv("ICECLAW_SOMBENCH_ROOT", "/nonexistent-iceclaw-sombench")
    for k in (
        "OPENAI_API_KEY",
        "ICECLAW_LLM_MODEL",
        "ICECLAW_LLM_PROVIDER",
        "ICECLAW_DATA_ROOT",
        "ICECLAW_RUNS_ROOT",
    ):
        monkeypatch.delenv(k, raising=False)
