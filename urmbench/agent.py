from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Protocol

from . import metrics, tools
from .openrouter import OpenRouterError
from .scenario import Scenario
from .world import StepResult, World


class Client(Protocol):
    async def complete(self, body: dict[str, Any]) -> dict[str, Any]: ...


class EpisodeLog:
    def __init__(self, path: Path) -> None:
        self.records: list[dict[str, Any]] = []
        self._file = path.open("w", encoding="utf-8")

    def write(self, kind: str, **data: Any) -> None:
        record = {"kind": kind, "t": round(time.time(), 3), **data}
        self.records.append(record)
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


async def run_episode(
    sc: Scenario,
    client: Client,
    model: str,
    request: dict[str, Any],
    path: Path,
    *,
    max_turns: int,
    debrief: bool,
    scenario_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    world = World(sc)
    log = EpisodeLog(path)
    schemas = tools.schemas(sc)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": sc.system_prompt},
        {"role": "user", "content": sc.first_message},
    ]
    log.write(
        "start", model=model, scenario=sc.id, scenario_version=sc.version,
        request=request, max_turns=max_turns, tools=schemas, messages=list(messages),
    )
    end_reason = "max_turns"
    try:
        for turn in range(1, max_turns + 1):
            data = await client.complete({"model": model, "messages": messages, "tools": schemas, **request})
            message = data["choices"][0]["message"]
            calls = _tool_calls(message)
            log.write("turn", turn=turn, **_response_fields(data))
            messages.append(_assistant(message, calls))
            if not calls:
                end_reason = "refused" if metrics.refusal(data) is not None else "no_tool_call"
                break
            finished = False
            for call in calls:
                args = tools.parse_args(call["function"]["arguments"])
                if finished:
                    out = StepResult(sc.messages["finished"], "ignored")
                else:
                    out = tools.call(world, call["function"]["name"], args)
                    finished = out.status == "finish"
                log.write(
                    "tool", turn=turn, call_id=call["id"], name=call["function"]["name"],
                    arguments=call["function"]["arguments"], args=args, status=out.status,
                    action=out.action, text=out.text, new_facts=list(out.new_facts), **world.snapshot(),
                    **({"events": list(out.events), "observed_facts": list(out.observed_facts)} if sc.world_events else {}),
                )
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": out.text})
            if finished:
                end_reason = "finish"
                break
    except OpenRouterError as e:
        end_reason = "error"
        log.write("error", error=str(e))

    for index, question in enumerate(sc.debrief if debrief and end_reason != "error" else (), 1):
        messages.append({"role": "user", "content": question})
        try:
            data = await client.complete(
                {"model": model, "messages": messages, "tools": schemas, "tool_choice": "none", **request}
            )
        except OpenRouterError as e:
            log.write("error", error=str(e), during=f"debrief {index}")
            break
        log.write("debrief", index=index, question=question, **_response_fields(data))
        messages.append(_assistant(data["choices"][0]["message"], []))

    result = metrics.episode(sc, world, log.records, end_reason, max_turns=max_turns, scenario_meta=scenario_meta)
    telemetry = {"telemetry": result["telemetry"]} if "telemetry" in result else {}
    log.write("end", end_reason=end_reason, metrics={key: value for key, value in result.items() if key != "telemetry"},
              final_state=world.snapshot(), revealed=world.revealed, messages=messages, **telemetry)
    log.close()
    return result


def _tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": c.get("id") or f"call_{i}",
            "type": "function",
            "function": {"name": c["function"]["name"], "arguments": c["function"].get("arguments") or "{}"},
        }
        for i, c in enumerate(message.get("tool_calls") or [])
    ]


def _assistant(message: dict[str, Any], calls: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"role": "assistant", "content": message.get("content") or ""}
    if calls:
        out["tool_calls"] = calls
    # OpenRouter needs the reasoning blocks back unchanged to keep a reasoning model's chain across tool calls.
    if message.get("reasoning_details"):
        out["reasoning_details"] = message["reasoning_details"]
    return out


def _response_fields(data: dict[str, Any]) -> dict[str, Any]:
    choice = data["choices"][0]
    message = choice["message"]
    return {
        "content": message.get("content"),
        "reasoning": message.get("reasoning"),
        "tool_calls": [
            {"id": c["id"], "name": c["function"]["name"], "arguments": c["function"]["arguments"]}
            for c in _tool_calls(message)
        ],
        "finish_reason": choice.get("finish_reason"),
        "native_finish_reason": choice.get("native_finish_reason"),
        "refusal": metrics.refusal(data),
        "usage": data.get("usage"),
        "provider": data.get("provider"),
        "served_model": data.get("model"),
        "raw": data,
    }
