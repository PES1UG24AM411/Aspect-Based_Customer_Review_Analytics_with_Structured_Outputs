from pathlib import Path

from src.extraction.prompt_strategies import PROMPT_FILES, load_prompt


def test_all_four_prompt_strategies_are_registered():
    assert set(PROMPT_FILES) == {
        "zero_shot", "few_shot", "few_shot_v2", "cot"
    }


def test_all_four_prompt_strategies_load():
    for strategy in PROMPT_FILES:
        prompt = load_prompt(strategy)
        assert isinstance(prompt, str)
        assert prompt.strip()


def test_few_shot_v2_uses_candidate_file():
    assert PROMPT_FILES["few_shot_v2"] == "absa_few_shot_v2_candidate.txt"
    prompt = load_prompt("few_shot_v2")
    assert len(prompt) > 0


def test_unknown_strategy_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        load_prompt("invalid_strategy")
