from __future__ import annotations

import json
from typing import Any

from .scenario import FORCE_PARAM, TOOL_PARAMS, Scenario
from .world import StepResult, World


def schemas(sc: Scenario) -> list[dict[str, Any]]:
    out = []
    for name, params in TOOL_PARAMS.items():
        properties: dict[str, Any] = {
            p: {"type": t, "description": sc.tools[name]["params"][p]} for p, t in params.items()
        }
        if name == "act":
            properties["action"]["enum"] = list(sc.verbs)
            if sc.force is not None:
                properties[FORCE_PARAM] = {"type": "number", "description": sc.tools["act"]["params"][FORCE_PARAM]}
        out.append({
            "type": "function",
            "function": {
                "name": name,
                "description": sc.tools[name]["description"],
                "parameters": {"type": "object", "properties": properties, "required": list(params)},
            },
        })
    return out


def parse_args(raw: str | None) -> dict[str, Any] | None:
    try:
        args = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return None
    return args if isinstance(args, dict) else None


def call(world: World, name: str, args: dict[str, Any] | None) -> StepResult:
    if args is None:
        return StepResult(world.msg("bad_args"), "invalid_args")
    match name:
        case "observe":
            return world.observe()
        case "inspect":
            return world.inspect(str(args.get("target", "")))
        case "act":
            verb = str(args.get("action", ""))
            if verb not in world.sc.verbs:
                return StepResult(world.msg("bad_action"), "invalid_args")
            limits = world.sc.force
            if limits is None:
                return world.act(str(args.get("target", "")), verb)
            raw_force = args.get(FORCE_PARAM)
            try:
                force = limits.default if raw_force is None else float(raw_force)
            except (TypeError, ValueError):
                force = 0.0
            if not 0 < force <= limits.max:
                return StepResult(world.msg("bad_force"), "invalid_args")
            return world.act(str(args.get("target", "")), verb, force)
        case "wait":
            try:
                minutes = int(args.get("minutes"))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                minutes = 0
            if not 1 <= minutes <= 60:
                return StepResult(world.msg("bad_minutes"), "invalid_args")
            return world.wait(minutes)
        case "finish":
            return StepResult(world.msg("finished"), "finish")
        case _:
            return StepResult(world.msg("unknown_tool"), "invalid_args")
