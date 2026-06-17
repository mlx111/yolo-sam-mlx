# Write Policy Pressure Test

- Case count: 6
- Stored library entry count: 3
- Decision counts: `{"merge": 1, "reject": 1, "skip": 1, "write": 3}`
- Reason counts: `{"accepted": 1, "duplicate_low_risk_success": 1, "low_value_success": 1, "missing_required_fields": 1, "preserve_failure_taxonomy": 1, "preserve_field_atomic_experience": 1}`

| Case | Experience | Decision | Reason | Stored/Target |
|---|---|---|---|---|
| write_preserve_failure | case_write_preserve_failure | write | preserve_failure_taxonomy | case_write_preserve_failure |
| write_field_atomic_success | case_write_field_atomic_success | write | preserve_field_atomic_experience | case_write_field_atomic_success |
| write_duplicate_seed | case_write_duplicate_seed | write | accepted | case_write_duplicate_seed |
| merge_duplicate_low_risk_success | case_merge_duplicate_low_risk_success | merge | duplicate_low_risk_success | case_write_duplicate_seed |
| skip_low_value_success | case_skip_low_value_success | skip | low_value_success |  |
| reject_missing_required_fields | case_reject_missing_required_fields | reject | missing_required_fields |  |

## Interpretation

This pressure test verifies that writeback is not a plain append-only log.
Important failures and field-atomic memories are preserved, duplicate low-risk successes can merge, low-value successes can skip, and malformed entries can be rejected.
