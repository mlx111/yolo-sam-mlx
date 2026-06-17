"""Render paper-ready evidence appendices from the claim evidence summary."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Markdown and LaTeX appendix tables from paper_evidence_summary.json."
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/memory/universal_pipeline_calibration_v1/paper_evidence_summary.json"),
        help="Input paper evidence summary JSON.",
    )
    parser.add_argument("--save-md", type=Path, required=True, help="Output Markdown appendix.")
    parser.add_argument("--save-tex", type=Path, required=True, help="Output LaTeX appendix.")
    parser.add_argument(
        "--max-metrics",
        type=int,
        default=6,
        help="Maximum metric items to show per claim in compact tables.",
    )
    return parser.parse_args()


def _load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("claims"), list):
        raise ValueError(f"Invalid evidence summary: {path}")
    return payload


def _compact_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _compact_metrics(metrics: dict[str, Any], max_items: int) -> str:
    if not metrics:
        return "n/a"
    parts: list[str] = []
    for index, (key, value) in enumerate(metrics.items()):
        if index >= max_items:
            parts.append(f"+{len(metrics) - max_items} more")
            break
        parts.append(f"{key}={_compact_value(value)}")
    return "; ".join(parts)


def _md_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _tex_escape(value: Any) -> str:
    text = str(value)
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
    return "".join(replacements.get(char, char) for char in text)


def _label_for_claim(index: int, claim: str) -> str:
    stem = re.sub(r"[^a-z0-9]+", "-", claim.lower()).strip("-")
    return f"C{index:02d}-{stem[:48]}"


def _status_counts(claims: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for claim in claims:
        status = str(claim.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def render_markdown(summary: dict[str, Any], max_metrics: int) -> str:
    claims = [claim for claim in summary.get("claims", []) if isinstance(claim, dict)]
    status_counts = _status_counts(claims)
    lines = [
        "# Paper Evidence Appendix",
        "",
        "This appendix maps paper claims to generated implementation reports and safe wording boundaries.",
        "Use `supported` claims as main-text claims. Treat `partial` claims as implementation support only, not validation evidence.",
        "",
        "## Summary",
        "",
        f"- Evidence summary: `{summary.get('report_dir', '')}`",
        f"- Claim count: {summary.get('claim_count', len(claims))}",
        f"- Supported claim count: {summary.get('supported_claim_count', status_counts.get('supported', 0))}",
        f"- Missing report count: {summary.get('missing_report_count', status_counts.get('missing_report', 0))}",
        f"- Status distribution: `{json.dumps(status_counts, ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Compact Evidence Table",
        "",
        "| ID | Status | Claim | Primary report | Key metrics | Safe wording | Avoid wording |",
        "|---|---|---|---|---|---|---|",
    ]
    for index, claim in enumerate(claims, start=1):
        claim_id = _label_for_claim(index, str(claim.get("claim", ""))).split("-", 1)[0]
        lines.append(
            "| "
            + " | ".join(
                [
                    claim_id,
                    _md_escape(claim.get("status", "")),
                    _md_escape(claim.get("claim", "")),
                    f"`{_md_escape(claim.get('primary_report', ''))}`",
                    _md_escape(_compact_metrics(claim.get("key_metrics") or {}, max_metrics)),
                    _md_escape(claim.get("safe_wording", "")),
                    _md_escape(claim.get("avoid_wording", "")),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Claim Cards", ""])
    for index, claim in enumerate(claims, start=1):
        label = _label_for_claim(index, str(claim.get("claim", "")))
        lines.extend(
            [
                f"### {label}",
                "",
                f"- Status: `{claim.get('status', '')}`",
                f"- Claim: {claim.get('claim', '')}",
                f"- Primary report: `{claim.get('primary_report', '')}`",
                f"- Safe wording: {claim.get('safe_wording', '')}",
                f"- Avoid wording: {claim.get('avoid_wording', '')}",
                f"- Key metrics: {_compact_metrics(claim.get('key_metrics') or {}, 999)}",
                "",
            ]
        )
    return "\n".join(lines)


def render_latex(summary: dict[str, Any], max_metrics: int) -> str:
    claims = [claim for claim in summary.get("claims", []) if isinstance(claim, dict)]
    status_counts = _status_counts(claims)
    lines = [
        r"\appendix",
        r"\section{Implementation Evidence Appendix}",
        (
            "This appendix maps claims to generated implementation reports. "
            "Supported claims may be used as main-text implementation claims; "
            "partial claims should be described as implementation support rather than validation evidence."
        ),
        "",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Claim-level evidence summary generated from implementation reports.}",
        r"\label{tab:claim-evidence-summary}",
        r"\begin{tabular}{p{0.06\linewidth}p{0.10\linewidth}p{0.36\linewidth}p{0.22\linewidth}p{0.18\linewidth}}",
        r"\hline",
        r"ID & Status & Claim & Primary report & Key metrics \\",
        r"\hline",
    ]
    for index, claim in enumerate(claims, start=1):
        claim_id = _label_for_claim(index, str(claim.get("claim", ""))).split("-", 1)[0]
        lines.append(
            " & ".join(
                [
                    _tex_escape(claim_id),
                    _tex_escape(claim.get("status", "")),
                    _tex_escape(claim.get("claim", "")),
                    r"\texttt{" + _tex_escape(claim.get("primary_report", "")) + "}",
                    _tex_escape(_compact_metrics(claim.get("key_metrics") or {}, max_metrics)),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\end{table*}",
            "",
            r"\paragraph{Summary.}",
            _tex_escape(
                f"Claim count={summary.get('claim_count', len(claims))}; "
                f"supported={summary.get('supported_claim_count', status_counts.get('supported', 0))}; "
                f"missing reports={summary.get('missing_report_count', status_counts.get('missing_report', 0))}; "
                f"status distribution={json.dumps(status_counts, ensure_ascii=False, sort_keys=True)}."
            ),
            "",
            r"\paragraph{Safe wording boundaries.}",
        ]
    )
    for index, claim in enumerate(claims, start=1):
        claim_id = _label_for_claim(index, str(claim.get("claim", ""))).split("-", 1)[0]
        lines.append(
            r"\noindent\textbf{"
            + _tex_escape(claim_id)
            + ".} Safe: "
            + _tex_escape(claim.get("safe_wording", ""))
            + " Avoid: "
            + _tex_escape(claim.get("avoid_wording", ""))
            + r"\\"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    summary = _load_summary(args.summary)
    args.save_md.parent.mkdir(parents=True, exist_ok=True)
    args.save_tex.parent.mkdir(parents=True, exist_ok=True)
    args.save_md.write_text(render_markdown(summary, args.max_metrics), encoding="utf-8")
    args.save_tex.write_text(render_latex(summary, args.max_metrics), encoding="utf-8")
    print(
        json.dumps(
            {
                "summary": str(args.summary),
                "save_md": str(args.save_md),
                "save_tex": str(args.save_tex),
                "claim_count": summary.get("claim_count"),
                "supported_claim_count": summary.get("supported_claim_count"),
                "missing_report_count": summary.get("missing_report_count"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
