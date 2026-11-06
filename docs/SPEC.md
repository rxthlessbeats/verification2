# Data and prompt contract

QERA schema version: `1.1`. Pydantic validates settings, records, overrides and
consumed API fields. Raw provider metadata is retained.

## Records

A JSONL record contains `traj_id`, `benchmark`, `domain`, `task_id`, `split`,
`solver`, `label`, ordered `segments`, `meta` and `hidden`.

| Field | Meaning |
|---|---|
| `solver` | Model, scaffold, origin and recorded reasoning availability |
| `label.correct` | Native outcome: true, false or null when unknown |
| `label.source/score/detail` | Grader and provenance |
| `segments[].seg_id` | Consecutive source order, starting at zero |
| `role_default/role_override` | Initial/effective role; read `segment.role` |
| `speaker/kind/step/source_id` | Source, analysis kind and original location |
| `text/call` | Recorded text and tool name/arguments |
| `spans` | Ordered, gap-free `[start,end)` role intervals covering the text |
| `call_role_override` | Call-argument role; defaults to the text role |
| `override_sources` | Per-field provenance; manual > classifier > rule |
| `ambiguous/note` | Conversion uncertainty and explanation |
| `meta/hidden` | Analysis/gold fields; never rendered into model input |

Q is requirements/policy, E is observed information, R is solver reasoning or
interpretation, and A is the candidate answer/action/artifact. Jev labels E/R;
Q/A stay fixed. Labels, hidden fields and adapter kinds are excluded from model
input. Neutral verifier headers contain only segment ID, speaker and step.

`native_reasoning` is full, summary, hidden or none, based on recorded reasoning,
not the model name. Overrides recalculate availability while preserving the
original declaration in `meta.native_reasoning_declared`. Incomplete output and
reordered tool results remain flagged; source order is preserved.

## Three roles

| Role | Input → output | Model | API |
|---|---|---|---|
| Solver | Native task/environment → fresh frozen episode and native outcome | `SOLVER_MODEL` | Chat Completions |
| Jev | Natural blocks without E/R hints → validated E/R overrides and records | `JEV_MODEL` | System One |
| Jev selector | Each E/R piece with the task and answer → useful, misleading or irrelevant (only with QJevA) | `JEV_MODEL` | System One |
| Verifier | Frozen QA/QEA/QRA/QERA/QJevA view → verdict and reason | `VERIFIER_MODEL` | Messages, else Chat Completions |

QJevA shows Q, A and only the E/R pieces Jev labelled useful; the selector
prompt is `selector` in `prompts.yaml`, and the experiment runs it only when
QJevA is among the conditions.

Each role names a model in `.env`; the model's endpoint for that API, its key
variable and limits come from `src/verification2/core/models.yaml`.

τ²'s simulated user and native judge inherit solver settings. They are benchmark
machinery, not extra research roles. Jev classifies E/R; it does not score tasks
or select evidence subsets.

Jev uses System One `state` plus typed E/R `choice` questions. Every question
requires exactly one valid answer; missing/extra IDs fail. The example uses
`typesafe-ai/jev`, `JEV_API_KEY` and the Vercel `/typesafe/v1/systemone` gateway.
A compatible TypeSafe model listed in `models.yaml` can replace it without
changing other roles.
No-target records pass through without a call. Valid partial batches publish with
an incomplete manifest; failed/dry-only batches publish no classified records.

Verifier responses must be JSON:

```json
{"verdict":"correct","reason":"Brief explanation"}
```

Only `correct`/`incorrect` verdicts are accepted. Malformed/truncated answers are
errors. Usage/cost is preserved from API responses, never estimated locally.

## Prompts

Edit [`prompts.yaml`](../src/verification2/core/prompts.yaml), or pass a versioned copy
with `--prompt-file` / `PROMPTS=...`:

- `verifier.instruction`: shared across views.
- `jev.instruction`, `jev.criteria.E/R`: classification definitions.
- `benchmarks.<name>.jev_guidance`: benchmark-specific interpretation.
- `benchmarks.<name>.solver_instruction`: null keeps native instructions.

Unknown keys/benchmarks, duplicate YAML keys, unsafe YAML and empty instructions
fail before calls. Native tasks, tools and grading stay intact.

## Overrides

Overrides target `traj_id`/`seg_id` and change `role`, `call_role` or `spans`.
Revalidation requires Q/A, valid spans and source order. The current experiment
uses no human labels.

Native policies, fixtures and frozen data are source records, not documentation;
keep their bytes unchanged.
