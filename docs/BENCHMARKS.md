# Benchmarks

The main experiment weights τ²-Bench, AgentDojo, AppWorld, BIRD and ScienceWorld equally. SWE-bench Verified is an optional runner. Source revisions are pinned
in [`native_sources.json`](../src/verification2/benchmarks/native_sources.json); task row
hashes and runtime locks are saved with each experiment.

| Benchmark | Input and evidence | Fresh scoring |
|---|---|---|
| [τ²-Bench](https://github.com/sierra-research/tau2-bench) | Policy, conversation, native tool results | Native reward/outcome evaluation |
| [SWE-bench Verified](https://github.com/SWE-bench/SWE-bench) | Issue, repository, command output, submitted patch | Official test harness |
| [AgentDojo](https://github.com/ethz-spylab/agentdojo) | Suite prompt, tool calls, tool results | Native utility checks (no prompt injections) |
| [AppWorld](https://github.com/stonybrooknlp/appworld) | Instruction, code executions, API outputs | State-based unit tests |
| [ScienceWorld](https://github.com/allenai/ScienceWorld) | Task text, commands, simulator replies | Task completion (`getCompleted`), not the 0–100 score |
| [BIRD Mini-Dev](https://bird-bench.github.io/) | Question, schema, read-only query results | SQLite result-set match; train for development, Mini-Dev held out |

AgentDojo runs the latest `v1.2.2` suites without injection attacks so the label
is pure task utility. AppWorld uses the official train+dev / test_normal split
and excludes `test_challenge`. See the
[protocol](FULL_RUN_PLAN.md) and [setup](FRESH_RUN.md).

## Candidate pilots

`scripts/candidate_pilot.py` runs a single evaluation-only task through the solver,
Jev and four verifier views. It selects the lowest SHA-256 of `42:` followed by
the question before reading the answer. Pilot files belong under `outputs/`.
GSM-Plus uses its v1 evaluation release and exact numeric grading for the selected
arithmetic item. AssistantBench uses the public dev release and the BrowserGym
0.14.3 scorer, retaining the fractional score and treating 1 as full credit.
GAIA uses the gated 2023 validation release, the attachment-free subset and
the official leaderboard scorer. Its official test labels are private.
The separate `--gaia-paper-example` option uses the authors' public Table 1
example with reworded question text. [GAIA paper](https://arxiv.org/pdf/2311.12983#page=20)

The pilot needs the project dependencies, NumPy, SciPy, PyArrow and
`huggingface_hub`. Run it in an isolated environment with `PYTHONPATH=src`
and the project's model settings. Pass the benchmark name (`gsm_plus`, `gaia`
or `assistantbench`) and `--output outputs/candidate-pilots/<name>`. GAIA
needs an authorized `HF_TOKEN` or `--gaia-file` pointing to official validation
Parquet. Search uses Brave when `BRAVE_SEARCH_API_KEY` is configured. Otherwise
it writes the solver's exact query to a `.request` file and waits for a sibling
`.json` array of public search hits with `title`, `url` and `snippet` fields.
Use only the query for retrieval and keep each receipt with the episode. After
16 research turns the solver gets a final submission turn. A missing submission
or failed service call leaves the episode unscored.

[GSM-Plus release and evaluation terms](https://huggingface.co/datasets/qintongli/GSM-Plus) · [GAIA access](https://huggingface.co/datasets/gaia-benchmark/GAIA) · [AssistantBench](https://assistantbench.github.io/)
