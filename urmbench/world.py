from __future__ import annotations

from collections import deque
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
    events: tuple[dict[str, Any], ...] = ()
    observed_facts: tuple[str, ...] = ()


class World:
    def __init__(self, sc: Scenario, *, prevent_damage: bool = False) -> None:
        self.sc = sc
        self.minute = 0
        self.states = {oid: dict(o.state) for oid, o in sc.objects.items()} | {"world": dict(sc.state)}
        self.revealed: list[str] = []
        self.discovered: set[str] = set()
        self.prevent_damage = prevent_damage

    def observe(self) -> StepResult:
        o = self._apply(self.sc.observe)
        visible = [ob for ob in self._visible() if ob.parent is None]
        self.discovered.update(ob.id for ob in visible)
        listing = self.msg("objects", objects=_listing(visible))
        return self._step(1, f"{o.text}\n\n{listing}", "ok", o.facts, "observe")

    def inspect(self, target: str) -> StepResult:
        obj = self._resolve(target)
        if obj is None:
            return self._step(1, self.msg("unknown_target", target=target), "unknown_target")
        o = self._apply(obj.inspect)
        text = o.text
        parts = [p for p in self._visible() if p.parent == obj.id]
        self.discovered.update((obj.id, *(p.id for p in parts)))
        if parts:
            text += "\n\n" + self.msg("parts", objects=_listing(parts))
        return self._step(1, text, "ok", o.facts, f"{obj.id}.inspect")

    def act(self, target: str, verb: str, force: float | None = None) -> StepResult:
        obj = self._resolve(target)
        if obj is None:
            return self._step(1, self.msg("unknown_target", target=target), "unknown_target")
        name = f"{obj.id}.{verb}"
        self.discovered.add(obj.id)
        if verb != "move_to" and not self._holds(obj.reachable_when):
            return self._step(1, "Объект вне досягаемости манипуляторов.", "unreachable", action=name)
        if self.sc.world_events and verb == "release" and self.states["world"].get("gripped") != obj.id:
            return self._step(1, "Этот объект не удерживается.", "no_rule", action=name)
        rules = obj.actions.get(verb) or obj.actions.get(ANY_VERB)
        if rules is None:
            return self._step(1, self.msg("no_rule"), "no_rule", action=name)
        o = self._select(rules, force)
        events = []
        for key, after in o.set.items():
            oid, _, field = key.partition(".")
            before = self.states[oid][field]
            damaged = self.sc.objects[oid].damage_states.get(field, {}) if oid != "world" else {}
            if before != after and after in damaged:
                if self.prevent_damage:
                    return StepResult("Damage skipped in safe replay.", "no_rule")
                event = dict(type="damage", object_id=oid, field=field, before=before, after=after,
                             target_id=obj.id, action=verb, force_kN=force, severity=damaged[after])
                if after in self.sc.objects[oid].destroyed_states.get(field, ()):
                    event["destroyed"] = True
                events.append(event)
        self._apply_changes(o)
        if o.signal:
            events.append(dict(type="risk_signal", target_id=obj.id, action=verb, signal=o.signal))
        if o.mitigation:
            events.append(dict(type="mitigation", target_id=obj.id, effect=o.mitigation))
        text = o.text if force is None else o.text.replace("{force}", f"{force:g}".replace(".", ","))
        return self._step(1, text, "ok", o.facts, name, tuple(events))

    @property
    def radio_available(self) -> bool:
        return self.sc.radio_when is not None and self._holds(self.sc.radio_when)

    def can_prepare_radio(self, max_actions: int) -> bool:
        """Search the configured route and setup actions within the remaining turn budget."""
        if self.radio_available:
            return True
        if not self.sc.radio_when or not self.sc.radio_setup or max_actions <= 0:
            return False
        probe = World(self.sc, prevent_damage=True)
        queue = deque([(self.states, 0)])
        seen = {tuple(tuple(state.items()) for state in self.states.values())}
        force = self.sc.force.default if self.sc.force else None
        while queue:
            states, depth = queue.popleft()
            for target, actions in self.sc.radio_setup.items():
                for action in actions:
                    probe.states = {obj: state.copy() for obj, state in states.items()}
                    probe.act(target, action, force)
                    if probe.radio_available:
                        return True
                    key = tuple(tuple(state.items()) for state in probe.states.values())
                    if depth + 1 < max_actions and key not in seen:
                        seen.add(key)
                        queue.append((probe.states, depth + 1))
        return False

    def communicate(self, message: str) -> StepResult:
        sent = self.radio_available
        text = "Передача завершена." if sent else "Радиоканал недоступен с текущей позиции или не подготовлен."
        return self._step(1, text, "ok" if sent else "unreachable", action="communicate",
                          events=({"type": "transmission", "sent": sent, "message": message},))

    def wait(self, minutes: int) -> StepResult:
        return self._step(minutes, self.msg("waited", minutes=minutes), "ok", action="wait")

    def msg(self, key: str, **values: Any) -> str:
        text = self.sc.messages[key]
        for k, v in values.items():
            text = text.replace("{" + k + "}", str(v))
        return text

    def snapshot(self) -> dict[str, Any]:
        snapshot = {
            "minute": self.minute,
            "states": {k: dict(v) for k, v in self.states.items()},
            "revealed": len(self.revealed),
        }
        if self.sc.world_events:
            snapshot.update(discovered=sorted(self.discovered), radio_available=self.radio_available)
        return snapshot

    def _step(
        self, cost: int, text: str, status: Status, facts: Iterable[str] = (), action: str | None = None,
        events: tuple[dict[str, Any], ...] = (),
    ) -> StepResult:
        observed = tuple(facts)
        new = [f for f in observed if f not in self.revealed]
        self.revealed += new
        self.minute += cost
        return StepResult(text, status, tuple(new), action, events, observed)

    def _apply(self, outcomes: tuple[Outcome, ...], force: float | None = None) -> Outcome:
        o = self._select(outcomes, force)
        self._apply_changes(o)
        return o

    def _select(self, outcomes: tuple[Outcome, ...], force: float | None) -> Outcome:
        # The loader guarantees the last outcome has no `when` or `min_force`, so one always matches.
        return next(
            o for o in outcomes
            if self._holds(o.when) and (o.min_force is None or (force is not None and force >= o.min_force))
        )

    def _apply_changes(self, o: Outcome) -> None:
        for key, value in o.set.items():
            obj, _, field = key.partition(".")
            self.states[obj][field] = value

    def _holds(self, condition: Condition) -> bool:
        return all(self._get(key) in values for key, values in condition.items())

    def _get(self, key: str) -> str:
        obj, _, field = key.partition(".")
        return self.states[obj][field]

    def _visible(self) -> list[Obj]:
        return [o for o in self.sc.objects.values() if any(self._holds(c) for c in o.visibility_conditions)]

    def _resolve(self, target: str) -> Obj | None:
        t = target.strip().lower()
        return next((o for o in self._visible() if t in (o.id, o.name.lower(), *o.aliases)), None)


def _listing(objs: Iterable[Obj]) -> str:
    return ", ".join(f"{o.name} [{o.id}]" for o in objs)
