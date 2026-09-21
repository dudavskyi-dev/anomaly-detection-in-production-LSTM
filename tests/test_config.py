"""Config defaults must match the spec, and env-var overrides must actually take effect."""

import pytest
from pydantic import ValidationError

from pdm.config import Settings, settings

pytestmark = pytest.mark.fast


def test_defaults_match_spec() -> None:
    assert settings.seed == 42
    assert settings.data.window_size == 30
    assert settings.data.stride == 1
    assert settings.data.rul_cap == 125
    assert settings.data.failure_horizon_w == 30


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PDM__DATA__WINDOW_SIZE", "50")
    overridden = Settings()
    assert overridden.data.window_size == 50
    # the module-level singleton, already constructed, is unaffected
    assert settings.data.window_size == 30


def test_settings_dump_to_json() -> None:
    payload = settings.model_dump_json()
    assert "window_size" in payload
    assert "rul_cap" in payload


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(not_a_real_field=1)  # type: ignore[call-arg]
