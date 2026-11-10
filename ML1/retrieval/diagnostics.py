from __future__ import annotations

from collections import defaultdict


def retrieval_warnings(q, final: list, issues: list[dict]) -> list[str]:
    warnings = []
    if not final:
        warnings.append("insufficient_retrieval_evidence")
    if q.period:
        versions = defaultdict(set)
        for row in final:
            meta = row.metadata
            group = meta["edition_group"]
            if group:
                versions[group].add(meta["temporal_status"])
        if any("applicable" not in states for states in versions.values()):
            warnings.append("applicable_edition_unconfirmed")
    lead = final[0] if final else None
    states = {
        row.metadata["temporal_status"] for row in final
        if row.metadata["doc_id"] == lead.metadata["doc_id"] or row.score >= lead.score * 0.64
    } if lead else set()
    if states & {
        "historical_version_missing", "historical_version_unverified",
        "period_overlaps_change", "snapshot_precedes_period",
    }:
        warnings.append("historical_or_date_coverage_unverified")
    groups = defaultdict(set)
    for row in final:
        if group := row.metadata["edition_group"]:
            groups[group].add(row.metadata["doc_id"])
    if any(len(versions) > 1 for versions in groups.values()):
        warnings.append("multiple_document_editions")
    if any(row.metadata["doc_type"] == "addendum" for row in final):
        warnings.append("check_addendum_applicability")
    if any(row.metadata.get("amendment_status") in {
        "amendment_trigger_unverified", "amendment_period_overlaps_change",
    } for row in final):
        warnings.append("amendment_trigger_date_unconfirmed")
    if any(item["issue"] == "quarantined_prompt_directive" for item in issues):
        warnings.append("source_prompt_directive_quarantined")
    if any(item["issue"] != "quarantined_prompt_directive" for item in issues):
        warnings.append("knowledge_base_parse_issues")
    return warnings
