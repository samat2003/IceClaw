import pytest

from iceclaw.config import ConfigError, load_dotenv_file, load_settings


def test_defaults():
    s = load_settings({})
    assert s.llm_provider == "openai"
    assert s.llm_model is None and not s.has_api_key
    assert str(s.runs_root) == "runs"


def test_env_values():
    s = load_settings(
        {
            "ICECLAW_LLM_MODEL": "m1",
            "OPENAI_API_KEY": "sk-secret",
            "ICECLAW_DATA_ROOT": "/d",
            "ICECLAW_RUNS_ROOT": "/r",
        }
    )
    assert s.llm_model == "m1" and s.has_api_key
    assert str(s.data_root) == "/d"


def test_secret_not_in_repr_or_redacted():
    s = load_settings({"OPENAI_API_KEY": "sk-secret-123"})
    assert "sk-secret-123" not in repr(s)
    assert "sk-secret-123" not in str(s.redacted())
    assert s.redacted()["openai_api_key_set"] is True


def test_missing_key_error_is_clear():
    with pytest.raises(ConfigError) as e:
        load_settings({"ICECLAW_LLM_MODEL": "m"}).require_live_llm()
    assert e.value.code == "MISSING_API_KEY"


def test_missing_model_and_provider():
    with pytest.raises(ConfigError) as e:
        load_settings({"OPENAI_API_KEY": "k"}).require_live_llm()
    assert e.value.code == "MISSING_LLM_MODEL"
    with pytest.raises(ConfigError) as e:
        load_settings({"ICECLAW_LLM_PROVIDER": "x", "OPENAI_API_KEY": "k"}).require_live_llm()
    assert e.value.code == "UNSUPPORTED_PROVIDER"


def test_live_ok():
    load_settings({"OPENAI_API_KEY": "k", "ICECLAW_LLM_MODEL": "m"}).require_live_llm()


def test_dotenv_skipped_by_flag(tmp_path):
    f = tmp_path / ".env"
    f.write_text("OPENAI_API_KEY=zzz\n")
    assert load_dotenv_file(f) is False  # conftest sets ICECLAW_SKIP_DOTENV
