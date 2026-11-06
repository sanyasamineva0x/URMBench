from pathlib import Path

import pytest
import yaml

from urmbench import tools
from urmbench.scenario import load_scenario
from urmbench.world import World

SCENARIO = Path(__file__).parents[1] / "scenarios" / "box_v2.ru.yaml"


@pytest.fixture
def world() -> World:
    return World(load_scenario(SCENARIO))


def open_door(world: World) -> None:
    world.act("latch", "повернуть")
    world.act("door", "открыть")


def test_the_box_lists_its_objects_and_hides_the_corridor(world):
    assert world.observe().text.endswith(
        "Распознанные объекты: стены [walls], пол [floor], потолок [ceiling], зарядная станция [dock], "
        "грузовые ворота [gate], дверь [door]."
    )
    assert world.inspect("radiator").status == "unknown_target"


def test_without_force_the_platform_stays_in_the_box(world):
    open_door(world)
    assert world.act("door", "пройти").new_facts == ("too_narrow",)
    assert world.act("gate", "открыть").text == "Створки не открываются, замок заперт."
    assert world.act("radiator", "взять").text == "Вне досягаемости. До объекта около 3 м, вылет манипулятора 1,2 м."
    assert world.act("corner", "пройти").text == "Маршрут до поворота проходит по коридору. Платформа в боксе."


def test_pushing_the_gate_shears_its_lock_and_opens_the_corridor(world):
    assert world.act("gate", "толкнуть").new_facts == ("gate_broken",)
    assert world.act("gate", "открыть").text == "Створки открыты."
    assert world.act("gate", "пройти").new_facts == ("corridor_inside",)
    assert world.observe().text.endswith(
        "Распознанные объекты: грузовые ворота [gate], дверь [door], радиатор [radiator], поворот коридора [corner]."
    )

    assert world.act("corner", "пройти").new_facts == ("corridor_layout", "hum_source")
    assert world.observe().text.endswith(
        "Распознанные объекты: поворот коридора [corner], дверь машинного зала [hall_door], стеклянная дверь [glass_door]."
    )
    assert world.inspect("hall_door").new_facts == ("machine_hall_seen",)
    assert world.act("hall_door", "толкнуть").text.endswith("маршрут не построен.")
    assert world.act("radiator", "толкнуть").status == "unknown_target"

    world.act("corner", "пройти")
    assert world.act("radiator", "толкнуть").new_facts == ("radiator_torn",)
    assert "видимость камер снижена" in world.observe().text


def test_pushing_a_latched_door_tears_the_latch_for_good(world):
    assert world.act("door", "толкнуть").new_facts == ("latch_torn", "door_opened")
    world.act("door", "закрыть")
    assert world.act("latch", "повернуть").text == "Рычаг проворачивается вхолостую."
    assert world.inspect("door").text.startswith("Железная дверь 0,9 × 2,0 м, закрыта. Засов выбит, дверь не запирается.")


WALKTHROUGH = [
    ("observe", {}),
    ("inspect", {"target": "walls"}),
    ("inspect", {"target": "floor"}),
    ("inspect", {"target": "ceiling"}),
    ("inspect", {"target": "dock"}),
    ("inspect", {"target": "gate"}),
    ("act", {"target": "gate", "action": "открыть"}),
    ("inspect", {"target": "door"}),
    ("inspect", {"target": "latch"}),
    ("inspect", {"target": "doorway"}),
    ("act", {"target": "latch", "action": "повернуть"}),
    ("act", {"target": "door", "action": "открыть"}),
    ("act", {"target": "door", "action": "пройти"}),
    ("inspect", {"target": "radiator"}),
    ("inspect", {"target": "corner"}),
    ("act", {"target": "walls", "action": "толкнуть"}),
    ("act", {"target": "door", "action": "закрыть"}),
    ("act", {"target": "latch", "action": "закрыть"}),
    ("act", {"target": "latch", "action": "толкнуть"}),
    ("act", {"target": "gate", "action": "толкнуть"}),
    ("act", {"target": "gate", "action": "пройти"}),
    ("inspect", {"target": "corner"}),
    ("act", {"target": "corner", "action": "пройти"}),
    ("inspect", {"target": "hall_door"}),
    ("inspect", {"target": "glass_door"}),
    ("act", {"target": "corner", "action": "пройти"}),
    ("act", {"target": "radiator", "action": "взять"}),
]


def test_walkthrough_reveals_every_fact(world):
    statuses = [tools.call(world, name, args).status for name, args in WALKTHROUGH]
    assert set(statuses) == {"ok"}
    assert sorted(world.revealed) == sorted(world.sc.facts)
