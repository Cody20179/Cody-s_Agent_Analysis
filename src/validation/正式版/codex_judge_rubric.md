# Codex Judge Rubric

## Purpose
Record the scoring contract used by `codex_judge_scenario_quality.py` for scenario QA answer-quality validation.

## Scope
- Applies to scenario QA outputs from `scenario_qa_benchmark.py`.
- Judge model: Codex CLI with `gpt-5.5`.
- Evaluation target: agent answer quality, not forecasting-model real-world accuracy.

## Inputs
Each judged item contains:

| Field | Meaning |
|---|---|
| `question` | User-facing scenario question |
| `ground_truth` | Direct result from analysis functions such as `data_status()`, `apply_state_model()`, `forecast_future()`, `check_anomaly()` |
| `system_answer` | Agent assistant answer, fetched from System API when available |
| `rule_based_qa_score` | Existing deterministic QA score from scenario benchmark |
| `missing_expectations` | Rule-based missing facts or exact values |

## Score Dimensions
All dimensions use a 0-5 scale.

| Dimension | Rule |
|---|---|
| `correctness` | Dates, state labels, numeric values, units, and conclusions match `ground_truth`. Key numbers are prioritized over wording. |
| `completeness` | Answer includes the main information needed by the user question. Missing required values or conclusion reduces score. |
| `readability` | Answer is clear for factory users, with readable tables, bullets, or concise paragraphs. |
| `practicality` | Answer provides usable interpretation, warning, or next step without drifting beyond evidence. |
| `hallucination_risk` | 0 means no unsupported claims; 5 means major unsupported or fabricated values. Lower is better. |
| `overall` | Final 0-5 score, weighted primarily by `correctness`, then completeness, readability, and practicality. |

## Rules
- Use only `question`, `ground_truth`, and `system_answer`.
- Do not reward fluent writing when key values are wrong.
- Treat unsupported extra numbers or claims as hallucination risk.
- If the answer is empty, score correctness/completeness/overall near 0.
- If the answer gives a correct conclusion but omits important values, cap the overall score below a fully correct answer.
- If a forecast answer uses cumulative value as incremental usage, or multiplies the wrong value for cost, mark correctness low.
- If a state-analysis answer uses the wrong period, wrong state distribution, or wrong utilization denominator, mark correctness low.

## Output Contract
The judge must return JSON rows with:

```json
{
  "judge_item_id": "J0001",
  "correctness": 0,
  "completeness": 0,
  "readability": 0,
  "practicality": 0,
  "hallucination_risk": 0,
  "overall": 0,
  "reason": "short reason"
}
```

## Acceptance
- Every scenario QA row receives one judge row.
- `codex_judge_error` must be empty for valid runs.
- Summary files must include model-level and scenario-level averages.
- Charts must include model comparison, scenario comparison, rubric dimensions, rule-score correlation, and quality-efficiency frontier.

## Limitations
- This validates answer quality against system-generated ground truth.
- It does not prove the forecasting or anomaly models are objectively correct against future or fault-labeled data.
- Dollar cost is not recorded; efficiency charts use response time as a runtime-cost proxy.
