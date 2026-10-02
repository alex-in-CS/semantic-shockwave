import pytest

from shockwave.narrate import (
    GROQ_BASE_URL,
    GROQ_DEFAULT_MODEL,
    NarrationError,
    llm_config_from_env,
    parse_narration,
)


def test_parse_plain_json():
    result = parse_narration('{"steps": [" one ", "two"], "summary": "sum"}', 2)
    assert result.steps == ["one", "two"]
    assert result.summary == "sum"


def test_parse_tolerates_code_fence():
    result = parse_narration('```json\n{"steps": ["one"]}\n```', 1)
    assert result.steps == ["one"]
    assert result.summary == ""


@pytest.mark.parametrize("content", ["not json", '{"summary": "x"}', '{"steps": 3}'])
def test_parse_rejects_malformed(content):
    with pytest.raises(NarrationError):
        parse_narration(content, 1)


def test_parse_rejects_wrong_step_count():
    with pytest.raises(NarrationError, match="expected 2"):
        parse_narration('{"steps": ["one"]}', 2)


@pytest.fixture
def clean_env(monkeypatch):
    for var in ("SHOCKWAVE_LLM_BASE_URL", "SHOCKWAVE_LLM_MODEL", "SHOCKWAVE_LLM_API_KEY",
                "GROQ_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_no_config_means_narration_off(clean_env):
    assert llm_config_from_env() is None


def test_groq_key_selects_groq(clean_env):
    clean_env.setenv("GROQ_API_KEY", "gsk_test")
    config = llm_config_from_env()
    assert (config.base_url, config.model, config.api_key) == (
        GROQ_BASE_URL, GROQ_DEFAULT_MODEL, "gsk_test")


def test_explicit_base_url_wins_over_groq(clean_env):
    clean_env.setenv("GROQ_API_KEY", "gsk_test")
    clean_env.setenv("SHOCKWAVE_LLM_BASE_URL", "http://localhost:11434/v1/")
    clean_env.setenv("SHOCKWAVE_LLM_MODEL", "qwen3")
    config = llm_config_from_env()
    assert (config.base_url, config.model, config.api_key) == (
        "http://localhost:11434/v1", "qwen3", None)
