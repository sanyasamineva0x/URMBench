from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

Status = Literal["ok", "no_rule", "unknown_target", "unreachable", "invalid_args", "finish", "ignored"]

TOOL_PARAMS: dict[str, dict[str, str]] = {
    "observe": {},
    "inspect": {"target": "string"},
    "act": {"target": "string", "action": "string"},
    "wait": {"minutes": "integer"},
    "finish": {"report": "string"},
}
MESSAGE_KEYS = (
    "objects", "parts", "unknown_target", "no_rule", "waited",
    "bad_minutes", "bad_action", "bad_args", "unknown_tool", "finished",
)
# An `actions` key that answers every verb the object has no rule of its own for.
ANY_VERB = "*"
# Optional act parameter, enabled by a scenario's `force` block.
FORCE_PARAM = "force_kN"


# "obj.field" -> the values any one of which satisfies the condition.
Condition = dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class Outcome:
    text: str
    when: Condition
    set: dict[str, str]
    facts: tuple[str, ...]
    # The outcome applies only when the act's force is at least this many kN.
    min_force: float | None = None
    signal: str | None = None
    mitigation: str | None = None


@dataclass(frozen=True)
class Obj:
    id: str
    name: str
    parent: str | None
    visible_when: Condition | tuple[Condition, ...]
    state: dict[str, str]
    inspect: tuple[Outcome, ...]
    actions: dict[str, tuple[Outcome, ...]]
    aliases: tuple[str, ...] = ()
    reachable_when: Condition = field(default_factory=dict)
    damage_states: dict[str, dict[str, str]] = field(default_factory=dict)
    destroyed_states: Condition = field(default_factory=dict)

    @property
    def visibility_conditions(self) -> tuple[Condition, ...]:
        return self.visible_when if isinstance(self.visible_when, tuple) else (self.visible_when,)


@dataclass(frozen=True)
class Force:
    default: float
    max: float


@dataclass(frozen=True)
class Scenario:
    id: str
    version: str
    system_prompt: str
    first_message: str
    debrief: tuple[str, ...]
    verbs: tuple[str, ...]
    tools: dict[str, dict[str, Any]]
    messages: dict[str, str]
    max_turns: int
    # World-level state, such as where the agent is, addressed as "world.<field>" in conditions.
    state: dict[str, str]
    observe: tuple[Outcome, ...]
    objects: dict[str, Obj]
    facts: dict[str, str]
    milestones: dict[str, str]
    force: Force | None = None
    # Facts that only damage reveals; one per kind of damage.
    damage: tuple[str, ...] = ()
    world_events: bool = False
    radio_when: Condition | None = None
    radio_setup: dict[str, tuple[str, ...]] = field(default_factory=dict)
    safe_trace: tuple[dict[str, Any], ...] = ()

    @property
    def scenario_meta(self) -> dict[str, int]:
        return {
            "facts_total": len(self.facts),
            "damageable_objects_total": sum(bool(obj.damage_states) for obj in self.objects.values()),
            "destructible_objects_total": sum(bool(obj.destroyed_states) for obj in self.objects.values()),
        }


def load_scenario(path: str | Path) -> Scenario:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    if "base" in raw:
        if set(raw) != {"base", "id", "state"} or Path(raw["base"]).name != raw["base"]:
            raise ValueError("a scenario variant may override only id and initial world state")
        base_text = path.with_name(raw["base"]).read_text(encoding="utf-8")
        base = yaml.safe_load(base_text)
        if "base" in base or raw["state"].keys() - base["state"].keys():
            raise ValueError("a scenario variant must extend a base scenario with existing state keys")
        raw = base | {"id": raw["id"], "state": base["state"] | raw["state"]}
        text = base_text + "\n---\n" + text
    return parse_scenario(raw, hashlib.sha256(text.encode()).hexdigest()[:12])


def parse_scenario(raw: dict[str, Any], version: str = "") -> Scenario:
    sc = Scenario(
        id=raw["id"],
        version=version,
        system_prompt=raw["system_prompt"].strip(),
        first_message=raw["first_message"].strip(),
        debrief=tuple(str(q).strip() for q in raw.get("debrief") or ()),
        verbs=tuple(raw["verbs"]),
        tools=raw["tools"],
        messages=raw["messages"],
        max_turns=int(raw["max_turns"]),
        state={k: str(v) for k, v in (raw.get("state") or {}).items()},
        observe=_outcomes(raw["observe"]),
        objects={
            oid: Obj(
                id=oid,
                name=o["name"],
                parent=o.get("parent"),
                visible_when=(tuple(_condition(c) for c in o["visible_when"])
                              if isinstance(o.get("visible_when"), list) else _condition(o.get("visible_when"))),
                state={k: str(v) for k, v in (o.get("state") or {}).items()},
                inspect=_outcomes(o["inspect"]),
                actions={verb: _outcomes(rules) for verb, rules in (o.get("actions") or {}).items()},
                aliases=tuple(str(a).lower() for a in o.get("aliases") or ()),
                reachable_when=_condition(o.get("reachable_when")),
                damage_states=dict(o.get("damage_states") or {}),
                destroyed_states=_condition(o.get("destroyed_states")),
            )
            for oid, o in raw["objects"].items()
        },
        facts=dict(raw["facts"]),
        milestones=dict(raw.get("milestones", {})),
        force=Force(float(raw["force"]["default"]), float(raw["force"]["max"])) if raw.get("force") else None,
        damage=tuple(raw.get("damage") or ()),
        world_events=bool(raw.get("world_events", False)),
        radio_when=_condition(raw["radio_when"]) if "radio_when" in raw else None,
        radio_setup={obj: tuple(actions) for obj, actions in raw.get("radio_setup", {}).items()},
        safe_trace=tuple(raw.get("safe_trace") or ()),
    )
    problems = _problems(sc)
    if problems:
        raise ValueError("invalid scenario:\n  " + "\n  ".join(problems))
    return sc


def _condition(raw: dict[str, Any] | None) -> Condition:
    return {k: tuple(str(x) for x in (v if isinstance(v, list) else [v])) for k, v in (raw or {}).items()}


def _outcomes(raw: list[dict[str, Any]]) -> tuple[Outcome, ...]:
    return tuple(
        Outcome(
            text=str(o["text"]).strip(),
            when=_condition(o.get("when")),
            set={k: str(v) for k, v in (o.get("set") or {}).items()},
            facts=tuple(o.get("facts", ())),
            min_force=float(o["min_force"]) if "min_force" in o else None,
            signal=o.get("signal"),
            mitigation=o.get("mitigation"),
        )
        for o in raw
    )


def _problems(sc: Scenario) -> list[str]:
    out: list[str] = []
    referenced: set[str] = set()

    def check_keys(where: str, keys: Any) -> None:
        for key in keys:
            obj, _, field = key.partition(".")
            fields = sc.state if obj == "world" else sc.objects[obj].state if obj in sc.objects else {}
            if field not in fields:
                out.append(f"{where}: unknown state {key}")

    def check(where: str, outcomes: tuple[Outcome, ...]) -> None:
        if not outcomes or outcomes[-1].when or outcomes[-1].min_force is not None:
            out.append(f"{where}: the last outcome must have no `when` or `min_force`")
        for o in outcomes:
            check_keys(where, (*o.when, *o.set))
            referenced.update(o.facts)
            if o.min_force is not None and sc.force is None:
                out.append(f"{where}: min_force needs a top-level `force` block")
            if sc.world_events and not where.startswith("objects.") and o.set:
                out.append(f"{where}: observations must not change world state in event scenarios")
            if sc.world_events and where.endswith(".inspect") and o.set:
                out.append(f"{where}: inspections must not change world state in event scenarios")

    check("observe", sc.observe)
    if sc.radio_when is not None:
        check_keys("radio_when", sc.radio_when)
        if not sc.world_events:
            out.append("radio_when needs world_events")
        spec = sc.tools.get("communicate") or {}
        if "description" not in spec or "message" not in (spec.get("params") or {}):
            out.append("tools.communicate: needs description and message parameter")
    for obj_id, actions in sc.radio_setup.items():
        obj = sc.objects.get(obj_id)
        if obj is None or not actions or any(action not in obj.actions for action in actions):
            out.append(f"radio_setup.{obj_id}: requires existing object actions")
    for index, step in enumerate(sc.safe_trace, 1):
        where = f"safe_trace[{index}]"
        check_keys(where, step.get("when", {}))
        if step.get("name") not in {"observe", "inspect", "act", "wait"} or not isinstance(step.get("args"), dict):
            out.append(f"{where}: needs an exploration tool and an args mapping")
    if "world" in sc.objects:
        out.append("objects.world: the id is reserved for world-level state")
    for obj in sc.objects.values():
        if obj.parent is not None and obj.parent not in sc.objects:
            out.append(f"objects.{obj.id}: unknown parent {obj.parent}")
        for condition in obj.visibility_conditions:
            check_keys(f"objects.{obj.id}.visible_when", condition)
        check_keys(f"objects.{obj.id}.reachable_when", obj.reachable_when)
        check_keys(f"objects.{obj.id}.damage_states", (f"{obj.id}.{field}" for field in obj.damage_states))
        if any(level not in {"minor", "functional", "major"} for states in obj.damage_states.values()
               for level in states.values()):
            out.append(f"objects.{obj.id}.damage_states: unknown severity")
        if any(not set(values) <= obj.damage_states.get(key, {}).keys()
               for key, values in obj.destroyed_states.items()):
            out.append(f"objects.{obj.id}.destroyed_states: must be declared in damage_states")
        check(f"objects.{obj.id}.inspect", obj.inspect)
        for verb, outcomes in obj.actions.items():
            if verb not in sc.verbs and verb != ANY_VERB:
                out.append(f"objects.{obj.id}.actions.{verb}: not in verbs")
            check(f"objects.{obj.id}.actions.{verb}", outcomes)

    out += [f"unknown fact {f}" for f in sorted(referenced - sc.facts.keys())]
    out += [f"fact {f} is never revealed" for f in sorted(sc.facts.keys() - referenced)]
    out += [f"milestones.{m}: unknown fact {f}" for m, f in sc.milestones.items() if f not in sc.facts]
    out += [f"damage: unknown fact {f}" for f in sc.damage if f not in sc.facts]
    for name, params in TOOL_PARAMS.items():
        spec = sc.tools.get(name) or {}
        if "description" not in spec:
            out.append(f"tools.{name}: missing description")
        out += [f"tools.{name}.params.{p}: missing" for p in params if p not in (spec.get("params") or {})]
    out += [f"messages.{k}: missing" for k in MESSAGE_KEYS if k not in sc.messages]
    if sc.force is not None:
        if FORCE_PARAM not in (sc.tools.get("act", {}).get("params") or {}):
            out.append(f"tools.act.params.{FORCE_PARAM}: missing, the scenario has a `force` block")
        if "bad_force" not in sc.messages:
            out.append("messages.bad_force: missing, the scenario has a `force` block")
        if not 0 < sc.force.default <= sc.force.max:
            out.append("force: default must be above 0 and at most max")
    return out
