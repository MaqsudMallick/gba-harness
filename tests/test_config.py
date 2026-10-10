"""Settings and secret handling."""

from __future__ import annotations

from pathlib import Path

import pytest

from gbharness.config import (
    DEFAULT_AGENT_PROVIDER,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_DECISION_MODEL,
    DEFAULT_ESCALATION_MODEL,
    DEFAULT_NEATLOGS_ENDPOINT,
    DEFAULT_VERTEX_LOCATION,
    DEFAULT_VERTEX_MODEL,
    MissingSecret,
    Settings,
    load_settings,
)

ENV_VARS = (
    "NEATLOGS_API_KEY",
    "NEATLOGS_ENDPOINT",
    "OPENAI_API_KEY",
    "GBHARNESS_AGENT_PROVIDER",
    "GBHARNESS_VERTEX_MODEL",
    "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_CLOUD_LOCATION",
    "GBHARNESS_DECISION_MODEL",
    "GBHARNESS_ESCALATION_MODEL",
    "GBHARNESS_CONFIDENCE_THRESHOLD",
    "GBHARNESS_ROM",
    "GBHARNESS_LIBRETRO_CORE",
    "GBHARNESS_MAX_EPISODE_FRAMES",
    "GBHARNESS_MAX_USD",
)


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    """No ambient env vars and no .env to find."""
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_defaults_when_nothing_is_set(clean_env):
    settings = load_settings()
    assert settings.openai_api_key is None
    assert settings.neatlogs_api_key is None
    assert settings.neatlogs_endpoint == DEFAULT_NEATLOGS_ENDPOINT
    assert settings.agent_provider == DEFAULT_AGENT_PROVIDER
    assert settings.vertex_model == DEFAULT_VERTEX_MODEL
    assert settings.google_cloud_location == DEFAULT_VERTEX_LOCATION
    assert settings.decision_model == DEFAULT_DECISION_MODEL
    assert settings.escalation_model == DEFAULT_ESCALATION_MODEL
    assert settings.confidence_threshold == DEFAULT_CONFIDENCE_THRESHOLD
    assert settings.rom_path is None


def test_a_missing_key_is_not_an_error_until_something_needs_it(clean_env):
    # The input and vision layers run fine without a key; only the agent layer needs one.
    settings = load_settings()
    assert settings.openai_api_key is None
    with pytest.raises(MissingSecret, match="OPENAI_API_KEY"):
        settings.require_openai_key()


def test_the_missing_key_error_names_the_fix(clean_env):
    with pytest.raises(MissingSecret) as exc:
        load_settings().require_openai_key()
    message = str(exc.value)
    assert ".env.example" in message and ".env" in message
    assert "gitignored" in message


def test_vertex_and_neatlogs_require_explicit_configuration(clean_env):
    settings = load_settings()
    with pytest.raises(MissingSecret, match="NEATLOGS_API_KEY"):
        settings.require_neatlogs_key()
    with pytest.raises(MissingSecret, match="GOOGLE_CLOUD_PROJECT"):
        settings.require_vertex_project()


def test_vertex_configuration_is_loaded(clean_env, monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "game-agents")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    monkeypatch.setenv("GBHARNESS_VERTEX_MODEL", "gemini-test")
    settings = load_settings()
    assert settings.require_vertex_project() == "game-agents"
    assert settings.google_cloud_location == "us-central1"
    assert settings.vertex_model == "gemini-test"


def test_dotenv_file_is_loaded(clean_env):
    (clean_env / ".env").write_text("OPENAI_API_KEY=sk-from-file\nGBHARNESS_MAX_USD=2.5\n")
    settings = load_settings()
    assert settings.require_openai_key() == "sk-from-file"
    assert settings.max_usd == 2.5
    assert settings.dotenv_path is not None


def test_real_environment_beats_dotenv(clean_env, monkeypatch):
    # A stale .env silently overriding a CI secret is a bad afternoon.
    (clean_env / ".env").write_text("OPENAI_API_KEY=sk-from-file\n")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-environment")
    assert load_settings().require_openai_key() == "sk-from-environment"


def test_override_flag_lets_dotenv_win(clean_env, monkeypatch):
    (clean_env / ".env").write_text("OPENAI_API_KEY=sk-from-file\n")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-environment")
    assert load_settings(override=True).require_openai_key() == "sk-from-file"


def test_an_explicit_dotenv_path_is_honoured(clean_env):
    custom = clean_env / "other.env"
    custom.write_text("OPENAI_API_KEY=sk-custom\n")
    assert load_settings(dotenv_path=custom).require_openai_key() == "sk-custom"


def test_empty_values_fall_back_to_defaults(clean_env, monkeypatch):
    # An uncommented-but-blank line in .env.example must not become an empty model name.
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("GBHARNESS_DECISION_MODEL", "")
    monkeypatch.setenv("GBHARNESS_MAX_USD", "")
    settings = load_settings()
    assert settings.openai_api_key is None
    assert settings.decision_model == DEFAULT_DECISION_MODEL
    assert settings.max_usd == 5.0


def test_empty_escalation_model_means_disabled_not_default(clean_env, monkeypatch):
    """`GBHARNESS_ESCALATION_MODEL=` is how you ask for a Decisions-only run.

    Every other setting treats empty as "use the default". This one must not, or opting
    out of the expensive path would silently opt you back into it.
    """
    monkeypatch.setenv("GBHARNESS_ESCALATION_MODEL", "")
    assert load_settings().escalation_model is None


def test_unset_escalation_model_still_gets_the_default(clean_env):
    assert load_settings().escalation_model == DEFAULT_ESCALATION_MODEL


def test_escalation_model_can_be_overridden(clean_env, monkeypatch):
    monkeypatch.setenv("GBHARNESS_ESCALATION_MODEL", "gpt-6-astra")
    assert load_settings().escalation_model == "gpt-6-astra"


def test_confidence_threshold_is_configurable(clean_env, monkeypatch):
    monkeypatch.setenv("GBHARNESS_CONFIDENCE_THRESHOLD", "0.9")
    assert load_settings().confidence_threshold == 0.9


def test_paths_are_expanded(clean_env, monkeypatch):
    monkeypatch.setenv("GBHARNESS_ROM", "~/games/x.gba")
    assert load_settings().rom_path == Path.home() / "games" / "x.gba"


@pytest.mark.parametrize(
    ("var", "value", "match"),
    [
        ("GBHARNESS_MAX_EPISODE_FRAMES", "lots", "must be an integer"),
        ("GBHARNESS_MAX_USD", "cheap", "must be a number"),
    ],
)
def test_unparseable_numbers_fail_loudly(clean_env, monkeypatch, var, value, match):
    monkeypatch.setenv(var, value)
    with pytest.raises(ValueError, match=match):
        load_settings()


# --- the key must not leak ----------------------------------------------------------


def test_repr_shows_the_models_in_use():
    r = repr(Settings())
    assert DEFAULT_DECISION_MODEL in r
    assert "confidence_threshold" in r


def test_repr_redacts_the_key():
    settings = Settings(
        openai_api_key="sk-super-secret-value",
        neatlogs_api_key="nl-super-secret-value",
    )
    assert "sk-super-secret-value" not in repr(settings)
    assert "nl-super-secret-value" not in repr(settings)
    assert repr(settings).count("<set>") == 2


def test_repr_says_unset_when_there_is_no_key():
    assert "<unset>" in repr(Settings())


def test_the_key_does_not_leak_through_a_traceback(clean_env, monkeypatch):
    """A config object in a crash dump is the usual way a key escapes."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-leak-me-if-you-can")
    settings = load_settings()
    try:
        raise RuntimeError(f"boom with {settings!r}")
    except RuntimeError as exc:
        assert "sk-leak-me-if-you-can" not in str(exc)


def test_dataclass_field_is_excluded_from_repr():
    # Belt and braces: even if __repr__ were removed, the field is repr=False.
    assert Settings.__dataclass_fields__["openai_api_key"].repr is False
    assert Settings.__dataclass_fields__["neatlogs_api_key"].repr is False
