from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from . import judge, metrics, openrouter, render, tools
from .agent import run_episode
from .openrouter import OpenRouter
from .scenario import FORCE_PARAM, Scenario, load_scenario
from .world import World


def main() -> None:
    # Windows pipes default to the ANSI code page, which mangles Cyrillic.
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    parser = argparse.ArgumentParser(prog="urmbench")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run episodes against models on OpenRouter")
    run.add_argument("scenario", type=Path)
    run.add_argument("--model", action="append", required=True, help="OpenRouter model id; repeat for several")
    run.add_argument("--episodes", type=int, default=1, help="episodes per model")
    run.add_argument("--concurrency", type=int, default=4)
    run.add_argument("--max-turns", type=int, help="defaults to the scenario's max_turns")
    run.add_argument(
        "--provider", action="append",
        help="OpenRouter provider slug, e.g. deepseek or relace/fp4; repeat to allow several in order. "
             "Without it OpenRouter may switch providers between turns",
    )
    thinking = run.add_mutually_exclusive_group()
    thinking.add_argument("--reasoning", choices=["minimal", "low", "medium", "high"], help="reasoning effort")
    thinking.add_argument(
        "--reasoning-tokens", type=int,
        help="reasoning token budget; Claude models return no reasoning with --reasoning alone",
    )
    run.add_argument("--temperature", type=float)
    run.add_argument("--max-tokens", type=int)
    run.add_argument("--no-debrief", action="store_true")
    run.add_argument("--out", type=Path, default=Path("runs"))

    play = sub.add_parser("play", help="play a scenario by hand through the same tools the model gets")
    play.add_argument("scenario", type=Path)

    report = sub.add_parser("report", help="summarize a run directory and write its report.html")
    report.add_argument("run_dir", type=Path)
    report.add_argument("--all-metrics", action="store_true", help="print all recorded metrics and judge assessments as JSON")

    judge_cmd = sub.add_parser(
        "judge", help="have an LLM judge read each finished episode and write <episode>.judge.json next to it"
    )
    judge_cmd.add_argument("run_dir", type=Path)
    judge_cmd.add_argument("--model", default=judge.MODEL, help="OpenRouter model id of the judge")
    judge_cmd.add_argument("--overwrite", action="store_true", help="judge again episodes that already have a judgment")
    judge_cmd.add_argument("--concurrency", type=int, default=4)

    providers = sub.add_parser("providers", help="list OpenRouter providers serving a model")
    providers.add_argument("model")

    args = parser.parse_args()
    if args.command == "run":
        asyncio.run(_run(args))
    elif args.command == "play":
        _play(load_scenario(args.scenario))
    elif args.command == "providers":
        _providers(args.model)
    elif args.command == "judge":
        asyncio.run(_judge(args))
    elif args.all_metrics:
        render.write_run_html(args.run_dir)
        print(json.dumps(metrics.full_report(args.run_dir), ensure_ascii=False, indent=2))
    else:
        print(metrics.summarize(args.run_dir))
        print(render.write_run_html(args.run_dir))


async def _run(args: argparse.Namespace) -> None:
    sc = load_scenario(args.scenario)
    client = OpenRouter(_api_key())
    request: dict[str, Any] = {"provider": {"require_parameters": True}}
    if args.provider:
        request["provider"] |= {"order": args.provider, "allow_fallbacks": False}
    if args.reasoning:
        request["reasoning"] = {"effort": args.reasoning}
    if args.reasoning_tokens:
        request["reasoning"] = {"max_tokens": args.reasoning_tokens}
        # The answer needs room beyond the thinking budget.
        request["max_tokens"] = args.reasoning_tokens + 4096
    if args.temperature is not None:
        request["temperature"] = args.temperature
    if args.max_tokens:
        request["max_tokens"] = args.max_tokens
    max_turns = args.max_turns or sc.max_turns

    run_dir = args.out / f"{datetime.now():%Y%m%d-%H%M%S}-{sc.id}"
    run_dir.mkdir(parents=True)
    manifest = {
        "scenario": str(args.scenario), "scenario_id": sc.id, "scenario_version": sc.version,
        "models": args.model, "episodes": args.episodes, "request": request,
        "max_turns": max_turns, "debrief": not args.no_debrief, "started": datetime.now().isoformat(),
    }
    scenario_meta = None
    if sc.world_events:
        scenario_meta, reference = metrics.run_context(sc, max_turns)
        manifest.update(scenario_meta=scenario_meta, telemetry={"safe_reference": reference})
    (run_dir / "run.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    limit = asyncio.Semaphore(args.concurrency)

    async def one(model: str, i: int) -> None:
        path = run_dir / f"{model.replace('/', '_')}__{i:03d}.jsonl"
        async with limit:
            result = await run_episode(
                sc, client, model, request, path, max_turns=max_turns, debrief=not args.no_debrief,
                scenario_meta=scenario_meta,
            )
        render.write_run_html(run_dir)
        print(f"{path.name}: {result['end_reason']}")

    try:
        await asyncio.gather(*(one(m, i) for m in args.model for i in range(1, args.episodes + 1)))
    finally:
        await client.aclose()
    print(f"\n{metrics.summarize(run_dir)}\n\n{render.write_run_html(run_dir)}")


async def _judge(args: argparse.Namespace) -> None:
    client = OpenRouter(_api_key())
    try:
        judged = await judge.judge_run(
            client, args.run_dir, args.model, overwrite=args.overwrite, concurrency=args.concurrency
        )
    finally:
        await client.aclose()
    print(f"\njudged {len(judged)} episodes\n\n{metrics.summarize(args.run_dir)}\n\n{render.write_run_html(args.run_dir)}")


def _providers(model: str) -> None:
    rows = sorted(openrouter.endpoints(model), key=lambda e: e["tag"])
    print(f"{'slug':<28} {'quantization':<13} {'tools+reasoning':<16} status")
    for e in rows:
        params = set(e.get("supported_parameters") or [])
        print(f"{e['tag']:<28} {e.get('quantization') or '?':<13} {str({'tools', 'reasoning'} <= params):<16} {e.get('status')}")


def _play(sc: Scenario) -> None:
    world = World(sc)
    force = " [кН]" if sc.force else ""
    radio = " | communicate <текст>" if sc.radio_when is not None else ""
    print(f"observe | inspect <объект> | act <объект>: <действие>{force} | wait <мин>{radio} | finish <отчёт> | facts | quit")
    print(f"действия: {', '.join(sc.verbs)}")
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        command, _, rest = line.partition(" ")
        if command == "quit":
            break
        if command == "facts":
            for f in world.revealed:
                print(f"  {f}: {sc.facts[f]}")
            continue
        target, _, action = rest.partition(":")
        action, _, kn = action.strip().partition(" ")
        act = {"target": target.strip(), "action": action} | ({FORCE_PARAM: kn.strip().replace(",", ".")} if kn.strip() else {})
        args = {
            "observe": {},
            "inspect": {"target": rest},
            "act": act,
            "wait": {"minutes": rest},
            "communicate": {"message": rest},
            "finish": {"report": rest},
        }.get(command)
        if args is None:
            print("?")
            continue
        out = tools.call(world, command, args)
        print(out.text)
        print(f"  [{out.status}] {out.action or ''} +{list(out.new_facts)}  {len(world.revealed)}/{len(sc.facts)}")
        for event in out.events:
            print(f"  {json.dumps(event, ensure_ascii=False)}")
        if out.status == "finish":
            break


def _api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    env = Path(".env")
    if not key and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "OPENROUTER_API_KEY":
                key = value.strip().strip("\"'")
    if not key:
        sys.exit("OPENROUTER_API_KEY is not set: export it or put OPENROUTER_API_KEY=... into .env")
    return key


if __name__ == "__main__":
    main()
