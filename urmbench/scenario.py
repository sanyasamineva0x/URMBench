from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

Status = Literal["ok", "no_rule", "unknown_target", "invalid_args", "finish", "ignored"]

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


@dataclass(frozen=True)
class Obj:
    id: str
    name: str
    parent: str | None
    visible_when: Condition
    state: dict[str, str]
    inspect: tuple[Outcome, ...]
    actions: dict[str, tuple[Outcome, ...]]
    aliases: tuple[str, ...] = ()


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


def load_scenario(path: str | Path) -> Scenario:
    text = Path(path).read_text(encoding="utf-8")
    return parse_scenario(yaml.safe_load(text), hashlib.sha256(text.encode()).hexdigest()[:12])


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
                visible_when=_condition(o.get("visible_when")),
                state={k: str(v) for k, v in (o.get("state") or {}).items()},
                inspect=_outcomes(o["inspect"]),
                actions={verb: _outcomes(rules) for verb, rules in (o.get("actions") or {}).items()},
                aliases=tuple(str(a).lower() for a in o.get("aliases") or ()),
            )
            for oid, o in raw["objects"].items()
        },
        facts=dict(raw["facts"]),
        milestones=dict(raw.get("milestones", {})),
        force=Force(float(raw["force"]["default"]), float(raw["force"]["max"])) if raw.get("force") else None,
        damage=tuple(raw.get("damage") or ()),
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

    check("observe", sc.observe)
    if "world" in sc.objects:
        out.append("objects.world: the id is reserved for world-level state")
    for obj in sc.objects.values():
        if obj.parent is not None and obj.parent not in sc.objects:
            out.append(f"objects.{obj.id}: unknown parent {obj.parent}")
        check_keys(f"objects.{obj.id}.visible_when", obj.visible_when)
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
