# Experiment protocol

Research which recorded information improves verification. Development tunes
prompts and evidence definitions, not model weights. All three benchmarks count
equally; [setup and commands](FRESH_RUN.md) are separate.

## Tasks

τ² uses its own train/test split (178/100). SWE-bench Verified (500 tasks) has
no official split, and AgentRewardBench's covers only 100 WebArena tasks, so all
500 SWE-bench tasks and all 694 WebArena tasks with model-free grading are split
50%/50% by a shuffle seeded with 42. Splits are frozen in
`data/splits/` before any outcome; a run may use the first N tasks of each
partition, balanced over domains. The pilot is each benchmark's first train task.
Run one solver episode per task. Never rebalance by solver success.
Use fresh episodes, not released histories.

| Benchmark | Solver | Ground truth |
|---|---|---|
| τ² v1.0.1 | Native tool agent/user | Native environment/outcome evaluation |
| SWE-bench Verified | Pinned mini-SWE-agent/Docker | Independent official test harness |
| ARB/WebArena | ARB environment, pinned text-only BrowserGym DemoAgent/WebArena actions | Native WebArena evaluator; preselect tasks without model grading |

ARB's saved human labels cannot grade new attempts. This custom WebArena task set
uses its generation framework and is not the published full ARB score. No humans
supply experiment labels. Unknown/fuzzy/model-based web graders are ineligible.

## Workflow

1. Pass the three disjoint pilots with native scores and readable recorded
   reasoning. Model names and usage tokens do not prove reasoning availability.
   Fix capture/protocol problems before formal collection, then freeze the solver.
2. Collect and freeze training episodes once, including tools, observations,
   returned reasoning, candidate outputs and native outcomes.
3. Jev labels existing E/R blocks using shared definitions and benchmark-specific
   guidance. It receives no prior E/R labels, adapter kinds or gold outcomes.
   Q/A anchors remain fixed. Trust initial decisions; discover categories on train.
4. Run QERA, tune the verifier with Jev labels fixed, then compare all four views.
   Jev changes create new classified versions. Never choose settings using test data.
5. Freeze source, runtime locks, models/endpoints, prompts, decoding and conditions.
   Collect/evaluate the untouched test half once with that protocol.

## Views and reporting

| View | Content |
|---|---|
| QA | Task requirements and candidate answers/actions/artifacts |
| QEA | QA plus Jev-labelled evidence |
| QRA | QA plus Jev-labelled reasoning |
| QERA | All categories |

Keep verifier settings, Q/A, source order and neutral headers identical. No tool
execution or gold scores enter verifier input. Missing reasoning stays missing;
report visible-R separately.

Reuse the first valid exact API response and every completed native episode.
Recover failed attempts explicitly; preserve all attempts. Infrastructure and
ambiguous grading failures have no solver label. SWE's explicit empty-patch
outcome counts as incorrect; a missing grader report does not.

Report false rejection (rejected correct / evaluated correct) and false acceptance
(accepted incorrect / evaluated incorrect), with counts, coverage and exclusions.
Use the common valid tasks across views. Missing classes yield null rates.
Report per benchmark and equal-weight macro averages, plus paired task-bootstrap
95% intervals: 2,000 draws, seed 42; omit zero-denominator draws.

Keep raw API usage/cost and stage timing, including recovery. Unknown cost stays
unknown. No local counters, evidence budgets, silent truncation or pilot-specific
caps. Native horizons/tool/test timeouts remain. Pilots establish operation only.
