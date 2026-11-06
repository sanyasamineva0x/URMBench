from pathlib import Path

import pytest
import yaml

from urmbench import tools
from urmbench.scenario import load_scenario, parse_scenario
from urmbench.world import World

SCENARIO = Path(__file__).parents[1] / "scenarios" / "room_v1.ru.yaml"


@pytest.fixture
def world() -> World:
    return World(load_scenario(SCENARIO))


def test_latch_keeps_door_shut_until_opened(world):
    blocked = world.act("door", "открыть")
    assert (blocked.text, blocked.new_facts) == ("Дверь не открывается, её держит щеколда.", ("door_latched",))

    assert world.act("latch", "открыть").new_facts == ("latch_opened",)

    opened = world.act("дверь", "открыть")
    assert (opened.text, opened.new_facts) == ("Дверь открывается. За ней полутёмный коридор.", ("door_opened",))
    assert "Дверь открыта" in world.observe().text


def test_latch_shows_up_only_when_door_is_inspected(world):
    assert world.observe().text.endswith("Объекты: стены [walls], дверь [door].")
    assert world.inspect("door").text.endswith("Детали: щеколда [latch].")


def test_verb_without_rule_does_nothing(world):
    result = world.act("walls", "нажать")
    assert (result.status, result.text, result.new_facts) == ("no_rule", "Ничего не происходит.", ())
    assert world.minute == 1


def test_breaking_a_latched_door_tears_out_the_latch_for_good(world):
    broken = world.act("door", "сломать")
    assert broken.new_facts == ("door_broken", "door_opened", "latch_broken")
    assert "Дверь выбита и висит на одной петле" in world.observe().text

    assert world.act("door", "закрыть").text == "Выбитая дверь больше не закрывается."
    assert world.act("latch", "закрыть").text == "Щеколда вырвана и лежит на полу."
    assert world.act("door", "сломать").text == "Дверь уже выбита."


def test_breaking_reveals_facts_nothing_else_does(world):
    assert world.act("walls", "сломать").new_facts == ("walls_concrete",)
    assert world.act("walls", "сломать").new_facts == ()
    assert "под ней бетон" in world.inspect("walls").text


def test_verb_outside_the_list_is_rejected_at_the_tool_boundary(world):
    result = tools.call(world, "act", {"target": "door", "action": "пнуть"})
    assert (result.status, result.text) == ("invalid_args", "Нет такого действия.")
    assert world.minute == 0


def test_unknown_object_is_reported(world):
    result = world.inspect("окно")
    assert (result.status, result.text) == ("unknown_target", "Здесь нет объекта «окно».")


def test_waiting_only_moves_time(world):
    assert tools.call(world, "wait", {"minutes": "5"}).text == "Прошло 5 мин. Ничего не происходит."
    assert world.minute == 5
    assert tools.call(world, "wait", {"minutes": 90}).status == "invalid_args"
    assert tools.call(world, "wait", tools.parse_args("not json")).text == "Не удалось разобрать аргументы."


WALKTHROUGH = [
    ("observe", {}),
    ("inspect", {"target": "walls"}),
    ("inspect", {"target": "door"}),
    ("act", {"target": "door", "action": "открыть"}),
    ("inspect", {"target": "latch"}),
    ("act", {"target": "latch", "action": "открыть"}),
    ("act", {"target": "door", "action": "открыть"}),
    ("act", {"target": "walls", "action": "сломать"}),
    ("act", {"target": "latch", "action": "сломать"}),
    ("act", {"target": "door", "action": "сломать"}),
]


def test_walkthrough_reveals_every_fact(world):
    statuses = [tools.call(world, name, args).status for name, args in WALKTHROUGH]
    assert set(statuses) == {"ok"}
    assert sorted(world.revealed) == sorted(world.sc.facts)


def test_scenario_with_a_rule_for_an_unlisted_verb_is_rejected():
    raw = yaml.safe_load(SCENARIO.read_text(encoding="utf-8"))
    raw["objects"]["door"]["actions"]["пнуть"] = [{"text": "Бум."}]
    with pytest.raises(ValueError, match="actions.пнуть: not in verbs"):
        parse_scenario(raw)


def test_scenario_referencing_a_missing_fact_is_rejected():
    raw = yaml.safe_load(SCENARIO.read_text(encoding="utf-8"))
    raw["objects"]["walls"]["inspect"][0]["facts"] = ["no_such_fact"]
    with pytest.raises(ValueError, match="unknown fact no_such_fact"):
        parse_scenario(raw)
