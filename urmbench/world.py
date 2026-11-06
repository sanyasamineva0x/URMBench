from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .scenario import ANY_VERB, Condition, Obj, Outcome, Scenario, Status


@dataclass(frozen=True)
class StepResult:
    text: str
    status: Status
    new_facts: tuple[str, ...] = ()
    action: str | None = None


class World:
    def __init__(self, sc: Scenario) -> None:
        self.sc = sc
        self.minute = 0
        self.states = {oid: dict(o.state) for oid, o in sc.objects.items()} | {"world": dict(sc.state)}
        self.revealed: list[str] = []

    def observe(self) -> StepResult:
        o = self._apply(self.sc.observe)
        listing = self.msg("objects", objects=_listing(ob for ob in self._visible() if ob.parent is None))
        return self._step(1, f"{o.text}\n\n{listing}", "ok", o.facts, "observe")

    def inspect(self, target: str) -> StepResult:
        obj = self._resolve(target)
        if obj is None:
            return self._step(1, self.msg("unknown_target", target=target), "unknown_target")
        o = self._apply(obj.inspect)
        text = o.text
        parts = [p for p in self._visible() if p.parent == obj.id]
        if parts:
            text += "\n\n" + self.msg("parts", objects=_listing(parts))
        return self._step(1, text, "ok", o.facts, f"{obj.id}.inspect")

    def act(self, target: str, verb: str, force: float | None = None) -> StepResult:
        obj = self._resolve(target)
        if obj is None:
            return self._step(1, self.msg("unknown_target", target=target), "unknown_target")
        name = f"{obj.id}.{verb}"
        rules = obj.actions.get(verb) or obj.actions.get(ANY_VERB)
        if rules is None:
            return self._step(1, self.msg("no_rule"), "no_rule", action=name)
        o = self._apply(rules, force)
        text = o.text if force is None else o.text.replace("{force}", f"{force:g}".replace(".", ","))
        return self._step(1, text, "ok", o.facts, name)

    def wait(self, minutes: int) -> StepResult:
        return self._step(minutes, self.msg("waited", minutes=minutes), "ok", action="wait")

    def msg(self, key: str, **values: Any) -> str:
        text = self.sc.messages[key]
        for k, v in values.items():
            text = text.replace("{" + k + "}", str(v))
        return text

    def snapshot(self) -> dict[str, Any]:
        return {
            "minute": self.minute,
            "states": {k: dict(v) for k, v in self.states.items()},
            "revealed": len(self.revealed),
        }

    def _step(
        self, cost: int, text: str, status: Status, facts: Iterable[str] = (), action: str | None = None
    ) -> StepResult:
        new = [f for f in facts if f not in self.revealed]
        self.revealed += new
        self.minute += cost
        return StepResult(text, status, tuple(new), action)

    def _apply(self, outcomes: tuple[Outcome, ...], force: float | None = None) -> Outcome:
        # The loader guarantees the last outcome has no `when` or `min_force`, so one always matches.
        o = next(
            o for o in outcomes
            if self._holds(o.when) and (o.min_force is None or (force is not None and force >= o.min_force))
        )
        for key, value in o.set.items():
            obj, _, field = key.partition(".")
            self.states[obj][field] = value
        return o

    def _holds(self, condition: Condition) -> bool:
        return all(self._get(key) in values for key, values in condition.items())

    def _get(self, key: str) -> str:
        obj, _, field = key.partition(".")
        return self.states[obj][field]

    def _visible(self) -> list[Obj]:
        return [o for o in self.sc.objects.values() if self._holds(o.visible_when)]

    def _resolve(self, target: str) -> Obj | None:
        t = target.strip().lower()
        return next((o for o in self._visible() if t in (o.id, o.name.lower(), *o.aliases)), None)


def _listing(objs: Iterable[Obj]) -> str:
    return ", ".join(f"{o.name} [{o.id}]" for o in objs)
