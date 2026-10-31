import pytest


@pytest.fixture(autouse=True)
def _never_read_dotenv(monkeypatch):
    """Tests must never open the user's .env or depend on ambient secrets."""
    monkeypatch.setenv("ICECLAW_SKIP_DOTENV", "1")
    for k in (
        "OPENAI_API_KEY",
        "ICECLAW_LLM_MODEL",
        "ICECLAW_LLM_PROVIDER",
        "ICECLAW_DATA_ROOT",
        "ICECLAW_RUNS_ROOT",
    ):
        monkeypatch.delenv(k, raising=False)
