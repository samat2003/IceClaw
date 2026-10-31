"""Central runtime settings. Secret values are never rendered in repr/str/logs."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_DATA_ROOT = "./data/processed"
DEFAULT_RUNS_ROOT = "./runs"
DEFAULT_PROVIDER = "openai"
SUPPORTED_PROVIDERS = ("openai",)

MISSING_API_KEY = "MISSING_API_KEY"


class ConfigError(RuntimeError):
    """Configuration problem with a machine-readable ``code``."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


@dataclass(frozen=True)
class Settings:
    data_root: Path
    runs_root: Path
    llm_provider: str = DEFAULT_PROVIDER
    llm_model: str | None = None
    openai_api_key: str | None = field(default=None, repr=False)

    @property
    def has_api_key(self) -> bool:
        return bool(self.openai_api_key)

    def require_live_llm(self) -> None:
        """Raise a clear error if live LLM investigation cannot run.

        Offline science, listing and verification never call this.
        """
        if self.llm_provider not in SUPPORTED_PROVIDERS:
            raise ConfigError(
                "UNSUPPORTED_PROVIDER",
                f"ICECLAW_LLM_PROVIDER={self.llm_provider!r}; supported: {SUPPORTED_PROVIDERS}.",
            )
        if not self.has_api_key:
            raise ConfigError(
                MISSING_API_KEY,
                "OPENAI_API_KEY is not set. Export it in your environment or local .env "
                "to run live investigations; offline tools and replay still work.",
            )
        if not self.llm_model:
            raise ConfigError(
                "MISSING_LLM_MODEL", "ICECLAW_LLM_MODEL is not set (a verified model ID)."
            )

    def redacted(self) -> dict[str, object]:
        """Loggable view; the key is reduced to a presence flag."""
        return {
            "data_root": str(self.data_root),
            "runs_root": str(self.runs_root),
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "openai_api_key_set": self.has_api_key,
        }


def load_dotenv_file(path: Path | None = None) -> bool:
    """Load ``.env`` from the current directory into ``os.environ`` (no override).

    Values are never printed. Skipped when ``ICECLAW_SKIP_DOTENV`` is set.
    """
    if os.environ.get("ICECLAW_SKIP_DOTENV"):
        return False
    from dotenv import load_dotenv

    target = path if path is not None else Path.cwd() / ".env"
    if not target.is_file():
        return False
    return load_dotenv(target, override=False)


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Build Settings from a mapping (default: ``os.environ``). Does not read ``.env``."""
    e = os.environ if env is None else env
    return Settings(
        data_root=Path(e.get("ICECLAW_DATA_ROOT") or DEFAULT_DATA_ROOT),
        runs_root=Path(e.get("ICECLAW_RUNS_ROOT") or DEFAULT_RUNS_ROOT),
        llm_provider=(e.get("ICECLAW_LLM_PROVIDER") or DEFAULT_PROVIDER).strip().lower(),
        llm_model=(e.get("ICECLAW_LLM_MODEL") or "").strip() or None,
        openai_api_key=(e.get("OPENAI_API_KEY") or "").strip() or None,
    )
