"""Convert memory ablation JSON reports into paper-friendly tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SUMMARY_COLUMNS = [
    ("Variant", "variant_label"),
    ("Runs", "run_count"),
    ("Success", "success_rate"),
    ("Changed", "candidate_changed_rate"),
    ("Risk Delta", "risk_score_delta_avg"),
    ("Critic Block", "critic_block_rate"),
    ("Critic Warn", "critic_warn_rate"),
    ("Repeat Fail", "repeated_failure_rate_avg"),
    ("Retrieval Delta", "retrieval_count_delta_avg"),
]

RUN_COLUMNS = [
    ("Variant", "variant_label"),
    ("Scenario", "scenario"),
    ("Condition", "condition"),
    ("Selected", "selected_candidate_id"),
    ("Changed", "candidate_changed"),
    ("Success", "success"),
    ("Risk Delta", "risk_score_delta"),
    ("Critic", "critic_status"),
    ("Calib Penalty", "selected_calibration_risk_penalty"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize memory ablation report JSON into Markdown and LaTeX tables.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--save-md", type=Path, default=None)
    parser.add_argument("--save-tex", type=Path, default=None)
    parser.add_argument("--title", default="Experience Memory Ablation")
    parser.add_argument("--caption", default="Ablation results for the experience-memory policy.")
    parser.add_argument("--label", default="tab:experience_memory_ablation")
    return parser.parse_args()


def _load_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"ablation report not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"ablation report must be a JSON object: {path}")
    if "summaries" not in payload:
        raise ValueError(f"ablation report missing summaries: {path}")
    return payload


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def _markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(label for label, _ in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(_fmt(row.get(key)) for _, key in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def _escape_latex(text: Any) -> str:
    value = _fmt(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def _latex_label(text: str) -> str:
    return "".join(char for char in str(text) if char.isalnum() or char in {":", "-", "_", "."})


def _latex_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]], *, caption: str, label: str) -> str:
    column_spec = "l" + "r" * (len(columns) - 1)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{_escape_latex(caption)}}}",
        rf"\label{{{_latex_label(label)}}}",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        " & ".join(_escape_latex(name) for name, _ in columns) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(_escape_latex(row.get(key)) for _, key in columns) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def build_markdown(report: dict[str, Any], *, title: str, caption: str, label: str) -> str:
    summaries = list(report.get("summaries") or [])
    runs = list(report.get("runs") or [])
    csv_rows = list(report.get("csv_rows") or [])
    per_run_rows = csv_rows if csv_rows else runs
    latex = _latex_table(summaries, SUMMARY_COLUMNS, caption=caption, label=label)
    metadata = [
        f"# {title}",
        "",
        "## Metadata",
        "",
        f"- Schema: `{report.get('schema_version', '')}`",
        f"- Config: `{report.get('config', '')}`",
        f"- Experience library: `{report.get('experience_library', '')}`",
        f"- Run count: `{report.get('run_count', len(per_run_rows))}`",
        f"- Execute selected: `{report.get('execute_selected', '')}`",
        "",
    ]
    sections = [
        *metadata,
        "## Summary",
        "",
        _markdown_table(summaries, SUMMARY_COLUMNS),
        "",
        "## Per-Run Selection",
        "",
        _markdown_table(per_run_rows, RUN_COLUMNS),
        "",
        "## LaTeX",
        "",
        "```latex",
        latex,
        "```",
        "",
    ]
    return "\n".join(sections)


def main() -> None:
    args = parse_args()
    report = _load_report(args.input)
    markdown = build_markdown(report, title=args.title, caption=args.caption, label=args.label)
    latex = _latex_table(list(report.get("summaries") or []), SUMMARY_COLUMNS, caption=args.caption, label=args.label)

    if args.save_md is not None:
        args.save_md.parent.mkdir(parents=True, exist_ok=True)
        args.save_md.write_text(markdown, encoding="utf-8")
    if args.save_tex is not None:
        args.save_tex.parent.mkdir(parents=True, exist_ok=True)
        args.save_tex.write_text(latex + "\n", encoding="utf-8")
    print(json.dumps({
        "summary_count": len(report.get("summaries") or []),
        "run_count": len(report.get("runs") or []),
        "save_md": str(args.save_md) if args.save_md else "",
        "save_tex": str(args.save_tex) if args.save_tex else "",
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
