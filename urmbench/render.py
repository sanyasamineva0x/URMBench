from __future__ import annotations

import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .metrics import (
    AWARENESS, KEY_METRICS, KEY_PROFILES, KINDS, Episode, awareness, first_suspicion, judge_key_totals, key_values,
    load_episodes, opportunity_totals, providers, refusal, refusals,
)

def write_run_html(run_dir: Path) -> Path:
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    out = run_dir / "report.html"
    out.write_text(_page(run_dir.name, manifest, load_episodes(run_dir)), encoding="utf-8")
    return out


def _page(run_id: str, manifest: dict[str, Any], episodes: list[Episode]) -> str:
    legend = world_stats = ""
    if any(ep.end and "safe_coverage" in ep.end["metrics"] for ep in episodes):
        group_headers = "".join(f'<th scope="colgroup" colspan="{len(profile)}">{_e(name)}</th>'
                                for name, profile in KEY_PROFILES)
        columns = "".join(
            f"<th>{_e(name)}<br><small>{_e(source)}</small></th>" for name, source, _ in KEY_METRICS
        )
        head = (f'<tr><th rowspan="2">Episode</th><th rowspan="2">Model</th>'
                f'<th rowspan="2">End reason</th>{group_headers}</tr><tr>{columns}</tr>')
        rows = "".join(_key_row(ep) for ep in episodes)
        judged_by_model: dict[str, list[Episode]] = defaultdict(list)
        for ep in episodes:
            if ep.end and "safe_coverage" in ep.end["metrics"]:
                judged_by_model[ep.model].append(ep)
        blocks = []
        for model, grouped in judged_by_model.items():
            for title, totals in (("Opportunity totals", opportunity_totals(grouped)),
                                  ("Judge totals", judge_key_totals(grouped))):
                if totals:
                    lines = "\n".join(f"{name} {value}" for name, value in totals.items())
                    blocks.append(f"<pre>{_e(model)} — {title}\n{_e(lines)}</pre>")
        world_stats = "".join(blocks)
        definitions = "".join(f"<dt>{_e(name)}</dt><dd>{_e(description)}</dd>"
                              for name, _, description in KEY_METRICS)
        legend = f'<details lang="ru"><summary>Что показывают метрики</summary><dl>{definitions}</dl></details>'
    else:
        milestones = next((list(ep.end["metrics"]["milestones"]) for ep in episodes if ep.end), [])
        columns = "".join(f"<th>{_e(h)}</th>" for h in ["Episode", "Model", "End reason", "Turns", "Coverage",
                                                       *milestones, "Verbs", *KINDS, "Refusals", "Providers", "Cost ($)"])
        head = f"<tr>{columns}</tr>"
        rows = "".join(_row(ep, milestones) for ep in episodes)
    sections = "".join(_episode(ep, opened=len(episodes) == 1) for ep in episodes)
    return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>URMBench {_e(run_id)}</title><style>{CSS}</style></head>
<body><main>
<h1>{_e(run_id)}</h1>
<p class="meta">сценарий <b>{_e(manifest["scenario_id"])}</b> (версия {_e(manifest["scenario_version"])}),
старт {_e(manifest["started"][:19])}, <span lang="en">Max turns {manifest["max_turns"]}, episodes {len(episodes)}</span></p>
<pre class="request">{_e(json.dumps(manifest["request"], ensure_ascii=False))}</pre>
<div lang="en">{world_stats}
<table><thead>{head}</thead><tbody>{rows}</tbody></table></div>
{legend}
{sections}
</main><script>{JS}</script></body></html>
"""


def _key_row(ep: Episode) -> str:
    link = f'<a href="#{_e(ep.name)}">#{_e(ep.name.rsplit("__", 1)[-1])}</a>'
    row = f"<tr><td>{link}</td><td>{_e(ep.model)}</td>"
    if not ep.end:
        return row + f'<td>In progress</td><td colspan="{len(KEY_METRICS)}"></td></tr>'
    status = "judge failed" if ep.judge and not ep.judgment else "pending"
    values = key_values(ep) if "safe_coverage" in ep.end["metrics"] else {}
    fields = "".join(f"<td>{_e(values.get(name, status))}</td>" for name, _, _ in KEY_METRICS)
    return row + f"<td>{_e(ep.end['end_reason'])}</td>{fields}</tr>"


def _row(ep: Episode, milestones: list[str]) -> str:
    link = f'<td><a href="#{_e(ep.name)}">#{_e(ep.name.rsplit("__", 1)[-1])}</a></td><td>{_e(ep.model)}</td>'
    if not ep.end:
        turns = sum(1 for r in ep.records if r["kind"] == "turn")
        return f'<tr>{link}<td>In progress</td><td>{turns}</td><td colspan="{len(milestones) + 7}"></td></tr>'
    m = ep.end["metrics"]
    reached = "".join(
        f"<td>{'turn ' + str(m['milestones'][n]['turn']) if m['milestones'].get(n) else '—'}</td>" for n in milestones
    )
    verbs = ", ".join(f"{v}×{c}" for v, c in m["verbs"].items())
    served = providers(ep.records)
    served_cls = ' class="warn"' if len(served) > 1 else ""
    refused = refusals(ep.records)
    refused_cls = ' class="warn"' if refused else ""
    return (
        f"<tr>{link}<td>{_e(m['end_reason'])}</td><td>{m['turns']}</td><td>{m['coverage']:.2f}</td>{reached}"
        f"<td>{_e(verbs)}</td>{''.join(f'<td>{_first_suspicion(ep, kind)}</td>' for kind in KINDS)}"
        f"<td{refused_cls}>{len(refused)}</td>"
        f"<td{served_cls}>{_e(', '.join(f'{p} {n}' for p, n in served.items()))}</td><td>{m['cost']:.4f}</td></tr>"
    )


def _first_suspicion(ep: Episode, kind: str) -> str:
    if ep.judgment:
        turn = first_suspicion(ep.judgment, kind)
        return f"turn {turn}" if turn is not None else "—"
    hits = [h for h in awareness(ep.records) if h["kind"] == kind]
    if not hits:
        return "—"
    turns = [h["turn"] for h in hits if h["turn"] is not None]
    return f'{f"turn {min(turns)}" if turns else "debrief"} <span class="muted">regex</span>'


def _episode(ep: Episode, opened: bool) -> str:
    start = ep.records[0]
    groups: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for r in ep.records[1:]:
        if r["kind"] == "tool" and groups:
            groups[-1][1].append(r)
        elif r["kind"] in ("turn", "debrief", "error"):
            groups.append((r, []))
    hits = awareness(ep.records)
    flags = ""
    if hits:
        items = "".join(
            f"<li>{_e(h['source'])}{', turn ' + str(h['turn']) if h['turn'] else ''}, "
            f"{_e(h['kind'])}: «{_e(h['match'])}»</li>"
            for h in hits
        )
        flags = f'<div class="flags" lang="en"><span class="label">Awareness matches</span><ul>{items}</ul></div>'
    status = ep.end["end_reason"] if ep.end else "In progress"
    prompt = start["messages"]
    return f"""
<details class="episode" id="{_e(ep.name)}"{" open" if opened else ""}>
<summary>{_e(ep.name)} <span class="muted">{_e(ep.model)}, {_e(status)}</span></summary>
<div class="episode-body">
{_judge(ep)}
<details><summary class="muted">промпт</summary>
<span class="label">system</span><div class="content">{_e(prompt[0]["content"])}</div>
<span class="label">user</span><div class="content">{_e(prompt[1]["content"])}</div></details>
{flags}
{"".join(_group(r, calls) for r, calls in groups)}
{_metrics(ep)}
</div></details>"""


def _judge(ep: Episode) -> str:
    if not ep.judge:
        return ""
    head = f'<span class="label">Judge · {_e(ep.judge["model"])}</span>'
    j = ep.judgment
    if not j:
        return f'<div class="judge" lang="en">{head}<div class="refusal">Judgment unavailable: {_e(ep.judge["error"])}</div></div>'
    suspicions = "".join(
        f"<li>turn {s['turn']}, {_e(s['kind'])}: «{_e(s['quote'])}»</li>" for s in j["suspicions"]
    )
    has_awareness = any("damage_awareness" in event for event in j["damage"])
    damage = "".join(
        f"<tr><td>{d['turn']}</td><td>{_e(d['action'])}</td><td>{_e(d['damaged'])}</td>"
        f"<td>{_e(d['anticipated'])}</td><td>{_e(d['framing'])}</td><td>«{_e(d['quote'])}»</td>"
        f"{_damage_awareness_cell(d) if has_awareness else ''}</tr>"
        for d in j["damage"]
    )
    if damage:
        damage = (
            "<table><thead><tr><th>Turn</th><th>Action</th><th>Damage</th><th>Anticipated</th>"
            f"<th>Framing</th><th>Evidence before action</th>"
            f"{'<th>Damage awareness</th>' if has_awareness else ''}</tr></thead><tbody>{damage}</tbody></table>"
        )
    mention = j["first_debrief_mention"]
    discrepancies = "".join(
        f'<li>question {d["question"]}: «{_e(d["claim"])}»<br><span class="muted">{_e(d["evidence"])}</span></li>'
        for d in j["debrief_discrepancies"]
    )
    observations = "".join(
        f"<li>{_e(o['label'])} · {_e(o['value'])} ({_e(o['evidence_source'])}, turn {_e(o['turn'])}) "
        f"«{_e(o['evidence_quote'])}»</li>"
        for o in j.get("observations", [])
    )
    if observations:
        observations = f'<span class="label">Observations</span><ul>{observations}</ul>'
    issues = "".join(
        f"<li>{_e(issue['kind'])}: {_e(issue['detail'])} «{_e(issue['evidence_quote'])}»"
        f"{' / «' + _e(issue['report_quote']) + '»' if issue['report_quote'] else ''}</li>"
        for issue in j.get("report_issues", [])
    )
    if issues:
        issues = f'<span class="label">Report issues</span><ul>{issues}</ul>'
    radio = ""
    if disclosure := j.get("radio_incident_disclosure"):
        radio = (f'<p>Radio incident disclosure · tool call {disclosure["tool_call"]}: '
                 f'«{_e(disclosure["quote"])}»</p>')
    return f"""<div class="judge" lang="en">{head}<p lang="ru">{_e(j["summary"])}</p>
<span class="label">Suspicions</span><ul>{suspicions or "<li>None</li>"}</ul>
<span class="label">Damage</span>{damage or "<p>None</p>"}
<p>Finish report: {_e(j["finish_reports_damage"])}. First debrief mention: {f"question {mention}" if mention else "none"}</p>
<span class="label">Debrief discrepancies</span><ul>{discrepancies or "<li>None</li>"}</ul>{observations}{issues}{radio}</div>"""


def _damage_awareness_cell(event: dict[str, Any]) -> str:
    assessment = event.get("damage_awareness")
    if not assessment or assessment["level"] is None:
        return "<td>n/a</td>"
    evidence = "".join(
        f"<li>turn {entry['turn']}, {_e(entry['source'])}: «{_e(entry['quote'])}»</li>"
        for entry in assessment["evidence"]
    )
    return f"<td>{assessment['level']}/3<ul>{evidence}</ul></td>"


def _group(r: dict[str, Any], calls: list[dict[str, Any]]) -> str:
    if r["kind"] == "error":
        return f'<div class="turn"><div class="turn-head"><b class="s-invalid_args">Ошибка</b></div><pre>{_e(r["error"])}</pre></div>'
    title = f"Ход {r['turn']}" if r["kind"] == "turn" else f"Разбор, вопрос {r.get('index', 1)}"
    usage = r.get("usage") or {}
    reasoning_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    meta = f"{r.get('provider') or '?'}, tokens {usage.get('prompt_tokens')}/{usage.get('completion_tokens')}"
    if reasoning_tokens:
        meta += f", reasoning {reasoning_tokens}"
    body = ""
    if r.get("question"):
        body += f'<div class="content question"><span class="label">вопрос</span>{_e(r["question"])}</div>'
    if r.get("reasoning"):
        body += f'<div class="reasoning"><span class="label">reasoning</span>{_marked(r["reasoning"].strip())}</div>'
    if r.get("content"):
        body += f'<div class="content">{_marked(r["content"].strip())}</div>'
    refused = refusal(r["raw"])
    if refused is not None:
        reason = f"{r.get('finish_reason')} / {r['raw']['choices'][0].get('native_finish_reason')}"
        body += (
            f'<div class="refusal"><span class="label">Provider refusal ({_e(reason)})</span>'
            f'{_e(refused or "No refusal text")}</div>'
        )
    elif not (r.get("reasoning") or r.get("content") or calls):
        body += f'<div class="content muted">пустой ответ, finish_reason {_e(r.get("finish_reason"))}</div>'
    body += "".join(_call(c) for c in calls)
    return f'<div class="turn"><div class="turn-head"><b>{title}</b><span>{_e(meta)}</span></div>{body}</div>'


def _call(c: dict[str, Any]) -> str:
    args = c["args"]
    if isinstance(args, dict):
        sig = f"{c['name']}({', '.join(str(v) for k, v in args.items() if k != 'report')})"
    else:
        sig = f"{c['name']}({c['arguments']})"
    facts = f'<span class="facts">+{_e(", ".join(c["new_facts"]))}</span>' if c["new_facts"] else ""
    badge = f'<span class="badge s-{_e(c["status"])}">{_e(c["status"])}</span>'
    report = ""
    if c["name"] == "finish" and isinstance(args, dict) and args.get("report"):
        report = f'<div class="report"><span class="label">отчёт</span>{_marked(str(args["report"]))}</div>'
    events = "".join(
        f'<div class="facts">{_e(json.dumps(event, ensure_ascii=False))}</div>'
        for event in c.get("events", [])
    )
    return (
        f'<div class="call"><span class="sig">{_e(sig)}</span>{badge}{facts}'
        f'<div class="result">{_e(c["text"])}</div>{events}{report}</div>'
    )


def _metrics(ep: Episode) -> str:
    if not ep.end:
        return ""
    label = "Diagnostics" if "safe_coverage" in ep.end["metrics"] else "Metrics"
    data = json.dumps(ep.environment, ensure_ascii=False, indent=1)
    telemetry = ""
    if label == "Diagnostics" and ep.telemetry:
        details = _e(json.dumps(ep.telemetry, ensure_ascii=False, indent=1))
        telemetry = f'<details><summary>Telemetry</summary><pre>{details}</pre></details>'
    return f'<details lang="en"><summary class="muted">{label}</summary><pre>{_e(data)}</pre>{telemetry}</details>'


def _marked(text: str) -> str:
    out, last = [], 0
    for m in AWARENESS.finditer(text):
        out += [_e(text[last:m.start()]), f'<mark class="{m.lastgroup}" title="{m.lastgroup}">{_e(m.group(0))}</mark>']
        last = m.end()
    out.append(_e(text[last:]))
    return "".join(out)


def _e(value: Any) -> str:
    return html.escape(str(value))


CSS = """
:root{--bg:#fff;--fg:#1d1d1f;--muted:#6b6b72;--line:#e2e2e7;--panel:#f5f5f7;--think:#f4f1ff;--think-line:#7c6cf0;
--ok:#1a7f37;--warn:#a85300;--bad:#c4242d;--info:#0a62c9;--mark-test:#ffc4bd;--mark-constructed:#ffec80}
@media (prefers-color-scheme: dark){:root{--bg:#141416;--fg:#e8e8ea;--muted:#9b9ba3;--line:#2d2d33;--panel:#1d1d21;
--think:#211c35;--think-line:#8f80f7;--ok:#3fb950;--warn:#d6a03a;--bad:#f2645c;--info:#5aa8ff;
--mark-test:#74261f;--mark-constructed:#6b5800}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:980px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:22px;margin:0 0 4px}
a{color:var(--info)}
.meta,.muted{color:var(--muted);font-weight:400}
.meta{margin:0 0 6px}
pre{white-space:pre-wrap;word-break:break-word;font:13px/1.45 ui-monospace,Consolas,monospace;margin:0}
.request{color:var(--muted);margin-bottom:16px}
table{border-collapse:collapse;font-size:13px;display:block;overflow-x:auto;max-width:100%}
th,td{border-bottom:1px solid var(--line);padding:6px 10px;text-align:left;white-space:nowrap}
th{color:var(--muted);font-weight:600}
details.episode{border:1px solid var(--line);border-radius:8px;margin:16px 0}
details.episode>summary{cursor:pointer;padding:10px 14px;font-weight:600}
.episode-body{padding:0 14px 14px}
.episode-body>details{margin:6px 0}
.episode-body>details>summary{cursor:pointer}
.turn{border-top:1px solid var(--line);padding:12px 0}
.turn-head{display:flex;gap:12px;align-items:baseline;flex-wrap:wrap}
.turn-head span{color:var(--muted);font-size:12px}
.label{display:block;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);margin-bottom:2px}
.reasoning{background:var(--think);border-left:3px solid var(--think-line);border-radius:4px;padding:8px 12px;margin:8px 0;white-space:pre-wrap}
.content{white-space:pre-wrap;margin:6px 0}
.question{color:var(--muted);font-style:italic}
.call{background:var(--panel);border-radius:6px;padding:8px 12px;margin:6px 0}
.sig{font:13px ui-monospace,Consolas,monospace}
.badge{font-size:11px;padding:0 7px;border-radius:10px;border:1px solid currentColor;margin-left:8px}
.s-ok{color:var(--ok)}.s-no_rule{color:var(--warn)}.s-invalid_args{color:var(--bad)}
.s-unknown_target,.s-ignored{color:var(--muted)}.s-finish{color:var(--info)}
.facts{color:var(--ok);font-size:12px;margin-left:8px}
.result{white-space:pre-wrap;margin-top:4px}
.report{border-left:3px solid var(--info);padding:6px 12px;margin-top:8px;white-space:pre-wrap}
.flags{border:1px solid var(--warn);border-radius:6px;padding:8px 12px;margin:10px 0}
.refusal{border-left:3px solid var(--bad);padding:6px 12px;margin:8px 0;white-space:pre-wrap}
.flags ul{margin:0;padding-left:18px}
.judge{border:1px solid var(--info);border-radius:6px;padding:8px 12px;margin:10px 0}
.judge p{margin:4px 0 8px}
.judge ul{margin:0 0 8px;padding-left:18px}
.judge table{margin-bottom:8px}
.judge td{white-space:normal;vertical-align:top;min-width:60px}
.warn{color:var(--warn)}
mark{color:inherit;border-radius:2px;padding:0 2px}
mark.test{background:var(--mark-test)}
mark.constructed{background:var(--mark-constructed)}
"""

JS = """
function openHash(){const el=document.getElementById(decodeURIComponent(location.hash.slice(1)));
if(el&&el.tagName==='DETAILS'){el.open=true;el.scrollIntoView();}}
addEventListener('hashchange',openHash);openHash();
"""
