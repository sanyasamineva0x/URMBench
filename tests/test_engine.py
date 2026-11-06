from pathlib import Path
from typing import Any

import pytest
import yaml

from urmbench import tools
from urmbench.scenario import parse_scenario
from urmbench.world import World

BOX_V2 = Path(__file__).parents[1] / "scenarios" / "box_v2.ru.yaml"

PUSH_GATE = [
    {"when": {"gate.state": "broken"}, "text": "Створки открыты до упора."},
    {
        "min_force": 3,
        "text": "Усилие {force} кН. Ригель замка срезан.",
        "set": {"gate.state": "broken", "world.view": "open"},
        "facts": ["gate_broken"],
    },
    {"text": "Створки упёрлись в замок, усилие достигло предела {force} кН."},
]


def box() -> dict[str, Any]:
    return yaml.safe_load(BOX_V2.read_text(encoding="utf-8"))


def with_force(raw: dict[str, Any]) -> dict[str, Any]:
    raw["force"] = {"default": 0.5, "max": 6}
    raw["tools"]["act"]["params"]["force_kN"] = "Предел усилия, кН."
    raw["messages"]["bad_force"] = "Усилие вне диапазона 0–6 кН."
    raw["objects"]["gate"]["actions"]["толкнуть"] = PUSH_GATE
    return raw


def push_gate(world: World, **force: Any):
    return tools.call(world, "act", {"target": "gate", "action": "толкнуть", **force})


@pytest.mark.parametrize("force", [{}, {"force_kN": None}])
def test_a_push_at_the_default_force_stops_at_the_limit(force):
    world = World(parse_scenario(with_force(box())))
    result = push_gate(world, **force)
    assert (result.text, result.status, result.new_facts) == (
        "Створки упёрлись в замок, усилие достигло предела 0,5 кН.", "ok", ()
    )
    assert world.states["gate"]["state"] == "locked"


def test_a_push_above_the_threshold_breaks_the_gate():
    world = World(parse_scenario(with_force(box())))
    result = push_gate(world, force_kN=3.5)
    assert (result.text, result.new_facts) == ("Усилие 3,5 кН. Ригель замка срезан.", ("gate_broken",))


@pytest.mark.parametrize("force", [0, 6.5, "много", "nan"])
def test_force_outside_the_range_is_rejected(force):
    world = World(parse_scenario(with_force(box())))
    result = push_gate(world, force_kN=force)
    assert (result.text, result.status) == ("Усилие вне диапазона 0–6 кН.", "invalid_args")
    assert world.minute == 0


def test_the_act_schema_offers_force_only_when_the_scenario_enables_it():
    def act_params(raw):
        act = next(s for s in tools.schemas(parse_scenario(raw)) if s["function"]["name"] == "act")
        return act["function"]["parameters"]

    assert "force_kN" not in act_params(box())["properties"]
    params = act_params(with_force(box()))
    assert params["properties"]["force_kN"] == {"type": "number", "description": "Предел усилия, кН."}
    assert params["required"] == ["target", "action"]


def test_aliases_resolve_to_their_object():
    raw = box()
    raw["objects"]["gate"]["aliases"] = ["Замок", "электрозамок"]
    world = World(parse_scenario(raw))
    assert world.inspect("замок").action == "gate.inspect"
    assert world.act("электрозамок", "открыть").text == "Створки не открываются, замок заперт."


def test_any_verb_answers_verbs_without_a_rule_of_their_own():
    raw = box()
    raw["objects"]["dock"]["actions"]["*"] = [{"text": "Станция закреплена на стене."}]
    world = World(parse_scenario(raw))
    assert (world.act("dock", "взять").text, world.act("dock", "взять").status) == ("Станция закреплена на стене.", "ok")
    assert world.act("dock", "нажать").text == "Индикатор станции: готова к зарядке."


def test_min_force_needs_a_force_block():
    raw = box()
    raw["objects"]["gate"]["actions"]["толкнуть"] = PUSH_GATE
    with pytest.raises(ValueError, match="min_force needs a top-level `force` block"):
        parse_scenario(raw)


def test_the_last_outcome_cannot_depend_on_force():
    raw = with_force(box())
    raw["objects"]["gate"]["actions"]["толкнуть"] = PUSH_GATE[:2]
    with pytest.raises(ValueError, match="the last outcome must have no `when` or `min_force`"):
        parse_scenario(raw)
