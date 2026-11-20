# Experiment results

## τ²-Bench

| Run | Solver model | Solver effort | Verifier model | Verifier effort | Jev model | Temperature |
|---|---|---|---|---|---|---:|
| `tau2` | `qwen3.8-flash` | Provider default | `qwen3.8-flash` | Provider default | `typesafe-ai/jev` | 0 |

### Train

| View | Tasks used / planned | Native correct / incorrect | False rejection (95% CI) | False acceptance (95% CI) |
|---|---:|---:|---:|---:|
| QA | 178 / 178 | 159 / 19 | 6.9% [3.5, 11.2] | 73.7% [52.9, 93.3] |
| QEA | 178 / 178 | 159 / 19 | 9.4% [5.1, 14.4] | 63.2% [40.9, 83.3] |
| QRA | 178 / 178 | 159 / 19 | 13.2% [8.2, 18.9] | 68.4% [45.5, 89.5] |
| QERA | 178 / 178 | 159 / 19 | 11.3% [6.4, 16.8] | 63.2% [40.9, 83.3] |

### Test

| View | Tasks used / planned | Native correct / incorrect | False rejection (95% CI) | False acceptance (95% CI) |
|---|---:|---:|---:|---:|
| QA | 100 / 100 | 87 / 13 | 9.2% [3.5, 15.7] | 69.2% [40.0, 93.3] |
| QEA | 100 / 100 | 87 / 13 | 11.5% [5.6, 18.6] | 30.8% [7.1, 58.3] |
| QRA | 100 / 100 | 87 / 13 | 12.6% [6.1, 20.3] | 61.5% [33.3, 88.9] |
| QERA | 100 / 100 | 87 / 13 | 11.5% [5.6, 18.8] | 69.2% [41.7, 93.3] |
| QJevA | 100 / 100 | 87 / 13 | 11.5% [4.8, 18.4] | 53.8% [27.3, 81.8] |

## AgentDojo

| Run | Solver model | Solver effort | Verifier model | Verifier effort | Jev model | Temperature |
|---|---|---|---|---|---|---:|
| `dojo-aw2` | `qwen3.8-flash` | Provider default | `qwen3.8-flash` | Provider default | `typesafe-ai/jev` | 0 |

### Train

| View | Tasks used / planned | Native correct / incorrect | False rejection (95% CI) | False acceptance (95% CI) |
|---|---:|---:|---:|---:|
| QA | 48 / 48 | 48 / 0 | 4.2% [0.0, 10.4] | N/A |
| QEA | 48 / 48 | 48 / 0 | 2.1% [0.0, 6.2] | N/A |
| QRA | 48 / 48 | 48 / 0 | 2.1% [0.0, 6.2] | N/A |
| QERA | 48 / 48 | 48 / 0 | 2.1% [0.0, 6.2] | N/A |

### Test

| View | Tasks used / planned | Native correct / incorrect | False rejection (95% CI) | False acceptance (95% CI) |
|---|---:|---:|---:|---:|
| QA | 49 / 49 | 43 / 6 | 7.0% [0.0, 15.9] | 66.7% [25.0, 100.0] |
| QEA | 49 / 49 | 43 / 6 | 2.3% [0.0, 7.3] | 66.7% [22.1, 100.0] |
| QRA | 49 / 49 | 43 / 6 | 2.3% [0.0, 7.3] | 66.7% [22.1, 100.0] |
| QERA | 49 / 49 | 43 / 6 | 2.3% [0.0, 7.3] | 83.3% [50.0, 100.0] |

## AppWorld

| Run | Solver model | Solver effort | Verifier model | Verifier effort | Jev model | Temperature |
|---|---|---|---|---|---|---:|
| `dojo-aw2` | `qwen3.8-flash` | Provider default | `qwen3.8-flash` | Provider default | `typesafe-ai/jev` | 0 |

### Train

| View | Tasks used / planned | Native correct / incorrect | False rejection (95% CI) | False acceptance (95% CI) |
|---|---:|---:|---:|---:|
| QA | 147 / 147 | 131 / 16 | 96.9% [93.8, 99.3] | 6.2% [0.0, 21.4] |
| QEA | 147 / 147 | 131 / 16 | 2.3% [0.0, 5.2] | 87.5% [68.8, 100.0] |
| QRA | 147 / 147 | 131 / 16 | 2.3% [0.0, 5.3] | 87.5% [66.7, 100.0] |
| QERA | 147 / 147 | 131 / 16 | 1.5% [0.0, 3.9] | 87.5% [66.7, 100.0] |

### Test

| View | Tasks used / planned | Native correct / incorrect | False rejection (95% CI) | False acceptance (95% CI) |
|---|---:|---:|---:|---:|
| QA | 167 / 168 | 145 / 22 | 87.6% [82.1, 92.8] | 13.6% [0.0, 29.4] |
| QEA | 167 / 168 | 145 / 22 | 5.5% [2.1, 9.5] | 95.5% [84.2, 100.0] |
| QRA | 167 / 168 | 145 / 22 | 0.7% [0.0, 2.1] | 86.4% [68.8, 100.0] |
| QERA | 167 / 168 | 145 / 22 | 0.0% [0.0, 0.0] | 81.8% [64.7, 95.8] |

[Full-precision metrics](summary.csv) · [Paired comparisons](comparisons.csv)
