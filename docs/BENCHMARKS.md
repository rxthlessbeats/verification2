# Benchmarks

The fresh experiment uses three benchmarks equally. Source revisions are pinned
in [`native_sources.json`](../src/verification2/benchmarks/native_sources.json); task row
hashes and runtime locks are saved with each experiment.

| Benchmark | Input and evidence | Fresh scoring |
|---|---|---|
| [τ²-Bench](https://github.com/sierra-research/tau2-bench) | Policy, conversation, native tool results | Native reward/outcome evaluation |
| [SWE-bench Verified](https://github.com/SWE-bench/SWE-bench) | Issue, repository, command output, submitted patch | Official test harness |
| [AgentRewardBench](https://github.com/McGill-NLP/agent-reward-bench) / WebArena | Goal, accessibility tree, browser actions | Native WebArena grading without model judges |

The ARB/WebArena task set is a custom native-scored study, not the published
human-labelled ARB score. Exclude fuzzy/model-based and unknown web evaluators
before sampling. See the [protocol](FULL_RUN_PLAN.md) and [setup](FRESH_RUN.md).
