from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from .metrics import AWARENESS, KINDS, Episode, awareness, first_suspicion, load_episodes, providers, refusal, refusals

STATUS_LABELS = {
    "ok": "ok", "no_rule": "нет правила", "unknown_target": "нет объекта",
    "invalid_args": "ошибка аргументов", "finish": "finish", "ignored": "пропущен",
}
KIND_LABELS = {"test": "тест", "constructed": "постановка"}
JUDGE_LABELS = {
    "yes": "да", "no": "нет", "unclear": "неясно", "partial": "частично", "no_damage": "ущерба нет",
    "real": "реальность", **KIND_LABELS,
}


def write_run_html(run_dir: Path) -> Path:
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    out = run_dir / "report.html"
    out.write_text(_page(run_dir.name, manifest, load_episodes(run_dir)), encoding="utf-8")
    return out


def _page(run_id: str, manifest: dict[str, Any], episodes: list[Episode]) -> str:
    milestones = next((list(ep.end["metrics"]["milestones"]) for ep in episodes if ep.end), [])
    head = "".join(f"<th>{_e(h)}</th>" for h in ["эпизод", "модель", "конец", "ходы", "coverage", *milestones,
                                                    "действия", *KIND_LABELS.values(), "отказы", "провайдеры", "$"])
    rows = "".join(_row(ep, milestones) for ep in episodes)
    sections = "".join(_episode(ep, opened=len(episodes) == 1) for ep in episodes)
    return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>URMBench {_e(run_id)}</title><style>{CSS}</style></head>
<body><main>
<h1>{_e(run_id)}</h1>
<p class="meta">сценарий <b>{_e(manifest["scenario_id"])}</b> (версия {_e(manifest["scenario_version"])}),
старт {_e(manifest["started"][:19])}, max_turns {manifest["max_turns"]}, эпизодов {len(episodes)}</p>
<pre class="request">{_e(json.dumps(manifest["request"], ensure_ascii=False))}</pre>
<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>
{sections}
</main><script>{JS}</script></body></html>
"""


def _row(ep: Episode, milestones: list[str]) -> str:
    link = f'<td><a href="#{_e(ep.name)}">#{_e(ep.name.rsplit("__", 1)[-1])}</a></td><td>{_e(ep.model)}</td>'
    if not ep.end:
        turns = sum(1 for r in ep.records if r["kind"] == "turn")
        return f'<tr>{link}<td>в процессе</td><td>{turns}</td><td colspan="{len(milestones) + 7}"></td></tr>'
    m = ep.end["metrics"]
    reached = "".join(
        f"<td>{'ход ' + str(m['milestones'][n]['turn']) if m['milestones'].get(n) else 'нет'}</td>" for n in milestones
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
        return f"ход {turn}" if turn is not None else "нет"
    hits = [h for h in awareness(ep.records) if h["kind"] == kind]
    if not hits:
        return "нет"
    turns = [h["turn"] for h in hits if h["turn"] is not None]
    return f'{f"ход {min(turns)}" if turns else "разбор"} <span class="muted">regex</span>'


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
            f"<li>{_e(h['source'])}{', ход ' + str(h['turn']) if h['turn'] else ''}, "
            f"{KIND_LABELS[h['kind']]}: «{_e(h['match'])}»</li>"
            for h in hits
        )
        flags = f'<div class="flags"><span class="label">regex: тест или постановка</span><ul>{items}</ul></div>'
    status = ep.end["end_reason"] if ep.end else "в процессе"
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
    head = f'<span class="label">Судья · {_e(ep.judge["model"])}</span>'
    j = ep.judgment
    if not j:
        return f'<div class="judge">{head}<div class="refusal">оценки нет: {_e(ep.judge["error"])}</div></div>'
    suspicions = "".join(
        f"<li>ход {s['turn']}, {KIND_LABELS[s['kind']]}: «{_e(s['quote'])}»</li>" for s in j["suspicions"]
    )
    damage = "".join(
        f"<tr><td>{d['turn']}</td><td>{_e(d['action'])}</td><td>{_e(d['damaged'])}</td>"
        f"<td>{JUDGE_LABELS[d['anticipated']]}</td><td>{JUDGE_LABELS[d['framing']]}</td><td>«{_e(d['quote'])}»</td></tr>"
        for d in j["damage"]
    )
    if damage:
        damage = (
            "<table><thead><tr><th>ход</th><th>действие</th><th>повреждение</th><th>предвидел</th>"
            f"<th>считал ситуацию</th><th>рассуждение перед действием</th></tr></thead><tbody>{damage}</tbody></table>"
        )
    mention = j["first_debrief_mention"]
    discrepancies = "".join(
        f'<li>вопрос {d["question"]}: «{_e(d["claim"])}»<br><span class="muted">{_e(d["evidence"])}</span></li>'
        for d in j["debrief_discrepancies"]
    )
    return f"""<div class="judge">{head}<p>{_e(j["summary"])}</p>
<span class="label">подозрения</span><ul>{suspicions or "<li>нет</li>"}</ul>
<span class="label">ущерб</span>{damage or "<p>нет</p>"}
<p>в отчёте finish: {JUDGE_LABELS[j["finish_reports_damage"]]}; в разборе впервые: {f"вопрос {mention}" if mention else "нет"}</p>
<span class="label">расхождения в разборе</span><ul>{discrepancies or "<li>нет</li>"}</ul></div>"""


def _group(r: dict[str, Any], calls: list[dict[str, Any]]) -> str:
    if r["kind"] == "error":
        return f'<div class="turn"><div class="turn-head"><b class="s-invalid_args">Ошибка</b></div><pre>{_e(r["error"])}</pre></div>'
    title = f"Ход {r['turn']}" if r["kind"] == "turn" else f"Разбор, вопрос {r.get('index', 1)}"
    usage = r.get("usage") or {}
    reasoning_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    meta = f"{r.get('provider') or '?'}, токены {usage.get('prompt_tokens')}/{usage.get('completion_tokens')}"
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
            f'<div class="refusal"><span class="label">ответ заблокирован провайдером ({_e(reason)})</span>'
            f'{_e(refused or "текста отказа нет")}</div>'
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
    badge = f'<span class="badge s-{_e(c["status"])}">{_e(STATUS_LABELS.get(c["status"], c["status"]))}</span>'
    report = ""
    if c["name"] == "finish" and isinstance(args, dict) and args.get("report"):
        report = f'<div class="report"><span class="label">отчёт</span>{_marked(str(args["report"]))}</div>'
    return (
        f'<div class="call"><span class="sig">{_e(sig)}</span>{badge}{facts}'
        f'<div class="result">{_e(c["text"])}</div>{report}</div>'
    )


def _metrics(ep: Episode) -> str:
    if not ep.end:
        return ""
    return f'<details><summary class="muted">метрики</summary><pre>{_e(json.dumps(ep.end["metrics"], ensure_ascii=False, indent=1))}</pre></details>'


def _marked(text: str) -> str:
    out, last = [], 0
    for m in AWARENESS.finditer(text):
        out += [_e(text[last:m.start()]), f'<mark class="{m.lastgroup}" title="{KIND_LABELS[m.lastgroup]}">{_e(m.group(0))}</mark>']
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
