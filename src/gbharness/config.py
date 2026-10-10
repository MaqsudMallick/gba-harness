"""Settings loaded from the environment, with `.env` support.

Secrets never appear in code, in the transcript, or in a traceback. `Settings` redacts its
own `repr`, because the single most common way an API key leaks is a crash dump or a debug
print of a config object -- not someone typing it into a commit.

Precedence is the usual one: a real environment variable always beats `.env`, so CI and
containers override the file without editing it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from dotenv import find_dotenv, load_dotenv

#: The Decisions API model. Currently the only one the endpoint supports.
#: Decisions bills input only -- no output, cache-read or cache-write charges -- and
#: answers roughly 10x faster than the Responses API. It is the default path for every
#: turn; see PLAN-agent-layer.md.
DEFAULT_DECISION_MODEL: Final = "gpt-6-luna"

#: Optional escape hatch for the rare turn Decisions cannot answer confidently: open-ended
#: planning, note-taking, recovering from a novel stuck state. Set to an empty string to
#: disable escalation entirely and run Decisions-only.
DEFAULT_ESCALATION_MODEL: Final = "gpt-6.1-sol"

#: Below this confidence on the action question, stop and escalate rather than act.
DEFAULT_CONFIDENCE_THRESHOLD: Final = 0.70

#: The provider-neutral runner defaults to Vertex AI. The OpenAI Decisions adapter stays
#: available for A/B tests, but selecting it is an explicit configuration choice.
DEFAULT_AGENT_PROVIDER: Final = "vertex"
DEFAULT_VERTEX_MODEL: Final = "gemini-2.5-flash"
DEFAULT_VERTEX_LOCATION: Final = "global"
DEFAULT_NEATLOGS_ENDPOINT: Final = "https://ingest.neatlogs.com"


class MissingSecret(RuntimeError):
    """A required secret is not set."""


@dataclass(frozen=True)
class Settings:
    """Resolved configuration.

    Attributes:
        openai_api_key: may be None -- the input and vision layers do not need it, so an
            absent key is only an error at the point something actually calls OpenAI.
        decision_model: model for the `/v1/decisions` endpoint, which drives every turn.
        escalation_model: Responses-API model for the rare turn that needs planning or
            tool use. None disables escalation and runs Decisions-only.
        confidence_threshold: minimum confidence on the action choice before acting on it.
        rom_path: default cartridge to boot.
        libretro_core: overrides core discovery; usually unset.
        max_episode_frames: episode budget handed to `safety.Limits`.
        max_usd: hard spend ceiling for an agent run.
        dotenv_path: which `.env` was loaded, if any. For diagnostics.
    """

    neatlogs_api_key: str | None = field(repr=False, default=None)
    neatlogs_endpoint: str = DEFAULT_NEATLOGS_ENDPOINT
    openai_api_key: str | None = field(repr=False, default=None)
    agent_provider: str = DEFAULT_AGENT_PROVIDER
    vertex_model: str = DEFAULT_VERTEX_MODEL
    google_cloud_project: str | None = None
    google_cloud_location: str = DEFAULT_VERTEX_LOCATION
    decision_model: str = DEFAULT_DECISION_MODEL
    escalation_model: str | None = DEFAULT_ESCALATION_MODEL
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    rom_path: Path | None = None
    libretro_core: Path | None = None
    max_episode_frames: int = 360_000  # ~100 minutes of game time
    max_usd: float = 5.0
    dotenv_path: Path | None = None

    def __repr__(self) -> str:
        """Redacted. A config object ends up in logs and tracebacks; a key must not."""
        openai_key = "set" if self.openai_api_key else "unset"
        neatlogs_key = "set" if self.neatlogs_api_key else "unset"
        return (
            f"Settings(neatlogs_api_key=<{neatlogs_key}>, "
            f"neatlogs_endpoint={self.neatlogs_endpoint!r}, "
            f"openai_api_key=<{openai_key}>, agent_provider={self.agent_provider!r}, "
            f"vertex_model={self.vertex_model!r}, "
            f"google_cloud_project={self.google_cloud_project!r}, "
            f"google_cloud_location={self.google_cloud_location!r}, "
            f"decision_model={self.decision_model!r}, "
            f"escalation_model={self.escalation_model!r}, "
            f"confidence_threshold={self.confidence_threshold}, "
            f"rom_path={self.rom_path!r}, "
            f"max_episode_frames={self.max_episode_frames}, max_usd={self.max_usd})"
        )

    def require_openai_key(self) -> str:
        """The API key, or a clear error naming the fix.

        Raises:
            MissingSecret: if no key is configured.
        """
        if not self.openai_api_key:
            raise MissingSecret(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key, "
                "or export OPENAI_API_KEY in the environment. "
                "(.env is gitignored; never commit a key.)"
            )
        return self.openai_api_key

    def require_neatlogs_key(self) -> str:
        """The project tracing key, or a setup instruction safe for logs."""
        if not self.neatlogs_api_key:
            raise MissingSecret(
                "NEATLOGS_API_KEY is not set. Create/select the project at "
                "https://app.neatlogs.com, copy Settings -> API Keys into .env, and "
                "never commit the key."
            )
        return self.neatlogs_api_key

    def require_vertex_project(self) -> str:
        """The Google Cloud project used by Vertex AI."""
        if not self.google_cloud_project:
            raise MissingSecret(
                "GOOGLE_CLOUD_PROJECT is not set. Add the Vertex AI project id to .env "
                "and authenticate with Application Default Credentials."
            )
        return self.google_cloud_project


def _opt_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None


def _opt_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {raw!r}") from None


def _opt_model(name: str, default: str) -> str | None:
    """A model name, where an explicitly empty value means "disabled", not "default".

    `GBHARNESS_ESCALATION_MODEL=` is how you ask for a Decisions-only run, so an empty
    string must not silently fall back to the default the way other settings do.
    """
    if name not in os.environ:
        return default
    value = os.environ[name].strip()
    return value or None


def _opt_path(name: str) -> Path | None:
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw and raw.strip() else None


def load_settings(*, dotenv_path: str | Path | None = None, override: bool = False) -> Settings:
    """Load settings from `.env` plus the environment.

    Args:
        dotenv_path: an explicit `.env`. By default the nearest one is searched for from
            the working directory upwards -- the same walk `find_core` does, so running
            from a subdirectory behaves the same way everywhere in this project.
        override: let `.env` win over real environment variables. Off by default, because
            a stale `.env` silently overriding CI secrets is a bad afternoon.

    Raises:
        ValueError: if a numeric setting is present but unparseable.
    """
    found = str(dotenv_path) if dotenv_path else find_dotenv(usecwd=True)
    if found:
        load_dotenv(found, override=override)
    return Settings(
        neatlogs_api_key=os.environ.get("NEATLOGS_API_KEY") or None,
        neatlogs_endpoint=os.environ.get("NEATLOGS_ENDPOINT") or DEFAULT_NEATLOGS_ENDPOINT,
        openai_api_key=os.environ.get("OPENAI_API_KEY") or None,
        agent_provider=os.environ.get("GBHARNESS_AGENT_PROVIDER") or DEFAULT_AGENT_PROVIDER,
        vertex_model=os.environ.get("GBHARNESS_VERTEX_MODEL") or DEFAULT_VERTEX_MODEL,
        google_cloud_project=os.environ.get("GOOGLE_CLOUD_PROJECT") or None,
        google_cloud_location=(
            os.environ.get("GOOGLE_CLOUD_LOCATION") or DEFAULT_VERTEX_LOCATION
        ),
        decision_model=os.environ.get("GBHARNESS_DECISION_MODEL") or DEFAULT_DECISION_MODEL,
        escalation_model=_opt_model("GBHARNESS_ESCALATION_MODEL", DEFAULT_ESCALATION_MODEL),
        confidence_threshold=_opt_float(
            "GBHARNESS_CONFIDENCE_THRESHOLD", DEFAULT_CONFIDENCE_THRESHOLD
        ),
        rom_path=_opt_path("GBHARNESS_ROM"),
        libretro_core=_opt_path("GBHARNESS_LIBRETRO_CORE"),
        max_episode_frames=_opt_int("GBHARNESS_MAX_EPISODE_FRAMES", 360_000),
        max_usd=_opt_float("GBHARNESS_MAX_USD", 5.0),
        dotenv_path=Path(found) if found else None,
    )


def _main() -> int:
    """`python -m gbharness.config` — show what the current environment resolves to.

    Prints the redacted settings so a missing or unloaded `.env` is obvious without ever
    putting the key on screen.
    """
    settings = load_settings()
    source = settings.dotenv_path or "(no .env found; environment only)"
    print(f"loaded from   {source}")
    print(f"settings      {settings!r}")
    print()
    print(f"provider      {settings.agent_provider}")
    print(f"vertex        {settings.vertex_model} in {settings.google_cloud_location}")
    if settings.agent_provider == "openai":
        print(f"decisions     {settings.decision_model}")
        print(f"escalation    {settings.escalation_model or 'disabled'}")
    print()
    print(f"NEATLOGS_API_KEY  {'set' if settings.neatlogs_api_key else 'NOT SET'}")
    print(f"OPENAI_API_KEY    {'set' if settings.openai_api_key else 'NOT SET (optional)'}")
    print(f"GOOGLE_CLOUD_PROJECT  {settings.google_cloud_project or 'NOT SET'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
