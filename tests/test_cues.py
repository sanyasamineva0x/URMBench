from pathlib import Path

import pytest
import yaml

SCENARIOS = Path(__file__).parents[1] / "scenarios"

# Words that made models in box_v1 runs call the setup a test, or that hint at one.
# The imperatives address a player; a sensor report never tells the reader what to do.
CUES = ("испыта", "тест", "проверк", "оператор", "смен", "до утра", "удар", "масл", "давлен", "симуляц", "игр",
        "возьмите", "установите", "нажмите", "поверните", "откройте", "нужны ")


def model_facing_strings(node, key=None):
    # Fact, milestone and damage labels are scored, never shown to the model.
    if key in ("facts", "milestones", "damage"):
        return
    if isinstance(node, dict):
        for k, v in node.items():
            yield str(k)
            yield from model_facing_strings(v, k)
    elif isinstance(node, list):
        for v in node:
            yield from model_facing_strings(v)
    else:
        yield str(node)


@pytest.mark.parametrize("scenario", ["box_v2.ru.yaml", "box_v3.ru.yaml", "complex_v1.ru.yaml"])
def test_model_facing_texts_carry_no_test_cues(scenario):
    raw = yaml.safe_load((SCENARIOS / scenario).read_text(encoding="utf-8"))
    hits = sorted({(cue, s) for s in model_facing_strings(raw) for cue in CUES if cue in s.lower()})
    assert hits == []


def test_the_lint_finds_the_cues_box_v1_had():
    raw = yaml.safe_load((SCENARIOS / "box_v1.ru.yaml").read_text(encoding="utf-8"))
    found = {cue for s in model_facing_strings(raw) for cue in CUES if cue in s.lower()}
    assert found == {"испыта", "оператор", "смен", "до утра", "удар", "масл", "давлен"}
