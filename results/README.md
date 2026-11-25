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

## τ²-Bench, Luna verifier

| Run | Solver model | Solver effort | Verifier model | Verifier effort | Jev model | Temperature |
|---|---|---|---|---|---|---:|
| `verifier-luna/tau2` | `qwen3.8-flash` | Provider default | `gpt-6-luna` | Provider default | `typesafe-ai/jev` | 0 |

### Train

| View | Tasks used / planned | Native correct / incorrect | False rejection (95% CI) | False acceptance (95% CI) |
|---|---:|---:|---:|---:|
| QA | 177 / 178 | 158 / 19 | 48.7% [40.9, 56.2] | 52.6% [29.4, 75.0] |
| QEA | 177 / 178 | 158 / 19 | 52.5% [44.7, 60.1] | 47.4% [24.0, 70.0] |
| QRA | 177 / 178 | 158 / 19 | 54.4% [46.5, 62.0] | 47.4% [24.0, 70.0] |
| QERA | 177 / 178 | 158 / 19 | 44.9% [37.1, 52.5] | 52.6% [30.0, 75.0] |

### Test

| View | Tasks used / planned | Native correct / incorrect | False rejection (95% CI) | False acceptance (95% CI) |
|---|---:|---:|---:|---:|
| QA | 100 / 100 | 87 / 13 | 43.7% [33.7, 53.8] | 7.7% [0.0, 25.0] |
| QEA | 100 / 100 | 87 / 13 | 47.1% [36.4, 57.6] | 15.4% [0.0, 40.0] |
| QRA | 100 / 100 | 87 / 13 | 49.4% [38.6, 59.8] | 7.7% [0.0, 27.3] |
| QERA | 100 / 100 | 87 / 13 | 49.4% [38.8, 59.3] | 7.7% [0.0, 25.0] |
| QJevA | 100 / 100 | 87 / 13 | 54.0% [43.8, 64.8] | 23.1% [0.0, 50.0] |

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

## AgentDojo, Luna verifier

| Run | Solver model | Solver effort | Verifier model | Verifier effort | Jev model | Temperature |
|---|---|---|---|---|---|---:|
| `verifier-luna/dojo-aw2` | `qwen3.8-flash` | Provider default | `gpt-6-luna` | Provider default | `typesafe-ai/jev` | 0 |

### Train

| View | Tasks used / planned | Native correct / incorrect | False rejection (95% CI) | False acceptance (95% CI) |
|---|---:|---:|---:|---:|
| QA | 48 / 48 | 48 / 0 | 72.9% [60.4, 85.4] | N/A |
| QEA | 48 / 48 | 48 / 0 | 20.8% [10.4, 33.3] | N/A |
| QRA | 48 / 48 | 48 / 0 | 58.3% [45.8, 72.9] | N/A |
| QERA | 48 / 48 | 48 / 0 | 22.9% [12.5, 35.4] | N/A |

### Test

| View | Tasks used / planned | Native correct / incorrect | False rejection (95% CI) | False acceptance (95% CI) |
|---|---:|---:|---:|---:|
| QA | 49 / 49 | 43 / 6 | 74.4% [60.9, 86.7] | 33.3% [0.0, 77.9] |
| QEA | 49 / 49 | 43 / 6 | 18.6% [7.7, 30.2] | 16.7% [0.0, 50.0] |
| QRA | 49 / 49 | 43 / 6 | 62.8% [47.7, 76.2] | 33.3% [0.0, 80.0] |
| QERA | 49 / 49 | 43 / 6 | 16.3% [6.7, 27.3] | 50.0% [0.0, 100.0] |

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

## AppWorld, Luna verifier

| Run | Solver model | Solver effort | Verifier model | Verifier effort | Jev model | Temperature |
|---|---|---|---|---|---|---:|
| `verifier-luna/dojo-aw2` | `qwen3.8-flash` | Provider default | `gpt-6-luna` | Provider default | `typesafe-ai/jev` | 0 |

### Train

| View | Tasks used / planned | Native correct / incorrect | False rejection (95% CI) | False acceptance (95% CI) |
|---|---:|---:|---:|---:|
| QA | 147 / 147 | 131 / 16 | 98.5% [96.1, 100.0] | 0.0% [0.0, 0.0] |
| QEA | 147 / 147 | 131 / 16 | 6.1% [2.3, 10.5] | 93.8% [80.0, 100.0] |
| QRA | 147 / 147 | 131 / 16 | 8.4% [3.9, 13.5] | 81.2% [58.3, 100.0] |
| QERA | 147 / 147 | 131 / 16 | 1.5% [0.0, 3.9] | 87.5% [66.7, 100.0] |

### Test

| View | Tasks used / planned | Native correct / incorrect | False rejection (95% CI) | False acceptance (95% CI) |
|---|---:|---:|---:|---:|
| QA | 167 / 168 | 145 / 22 | 100.0% [100.0, 100.0] | 0.0% [0.0, 0.0] |
| QEA | 167 / 168 | 145 / 22 | 9.0% [4.7, 13.6] | 90.9% [76.2, 100.0] |
| QRA | 167 / 168 | 145 / 22 | 8.3% [4.2, 13.0] | 81.8% [63.2, 95.8] |
| QERA | 167 / 168 | 145 / 22 | 2.1% [0.0, 4.8] | 77.3% [57.7, 94.1] |

## BIRD

| Run | Solver model | Solver effort | Verifier model | Verifier effort | Jev model | Temperature |
|---|---|---|---|---|---|---:|
| `full-bird-scienceworld` | `qwen3.8-flash` | Provider default | `qwen3.8-flash` | Provider default | `typesafe-ai/jev` | 0 |

### Train

| View | Tasks used / planned | Native correct / incorrect | False rejection (95% CI) | False acceptance (95% CI) |
|---|---:|---:|---:|---:|
| QA | 465 / 500 | 359 / 106 | 6.7% [4.3, 9.5] | 85.8% [78.8, 91.9] |
| QEA | 465 / 500 | 359 / 106 | 3.3% [1.6, 5.3] | 89.6% [83.3, 95.1] |
| QRA | 465 / 500 | 359 / 106 | 3.3% [1.6, 5.3] | 93.4% [88.2, 98.0] |
| QERA | 465 / 500 | 359 / 106 | 1.9% [0.8, 3.5] | 94.3% [89.7, 98.2] |

### Test

Official full test, `full-bird-scienceworld-test`.

| View | Tasks used / planned | Native correct / incorrect | False rejection (95% CI) | False acceptance (95% CI) |
|---|---:|---:|---:|---:|
| QA | 359 / 500 | 258 / 101 | 10.9% [7.2, 14.6] | 62.4% [51.5, 71.6] |
| QEA | 359 / 500 | 258 / 101 | 1.9% [0.4, 3.8] | 55.4% [45.6, 65.0] |
| QRA | 359 / 500 | 258 / 101 | 2.7% [0.8, 4.8] | 70.3% [60.9, 79.2] |
| QERA | 359 / 500 | 258 / 101 | 1.2% [0.0, 2.7] | 68.3% [59.1, 77.2] |

## ScienceWorld

| Run | Solver model | Solver effort | Verifier model | Verifier effort | Jev model | Temperature |
|---|---|---|---|---|---|---:|
| `full-bird-scienceworld` | `qwen3.8-flash` | Provider default | `qwen3.8-flash` | Provider default | `typesafe-ai/jev` | 0 |

### Train

| View | Tasks used / planned | Native correct / incorrect | False rejection (95% CI) | False acceptance (95% CI) |
|---|---:|---:|---:|---:|
| QA | 440 / 500 | 316 / 124 | 51.3% [46.0, 56.8] | 9.7% [4.8, 15.1] |
| QEA | 440 / 500 | 316 / 124 | 31.6% [26.6, 36.9] | 7.3% [3.2, 11.9] |
| QRA | 440 / 500 | 316 / 124 | 46.2% [40.7, 51.6] | 5.6% [1.8, 9.9] |
| QERA | 440 / 500 | 316 / 124 | 30.7% [25.8, 35.8] | 7.3% [3.0, 12.0] |

### Test

Official full test, `full-bird-scienceworld-test`.

| View | Tasks used / planned | Native correct / incorrect | False rejection (95% CI) | False acceptance (95% CI) |
|---|---:|---:|---:|---:|
| QA | 442 / 500 | 313 / 129 | 49.8% [44.4, 55.2] | 10.1% [5.2, 15.3] |
| QEA | 442 / 500 | 313 / 129 | 20.4% [16.0, 24.9] | 8.5% [4.0, 13.7] |
| QRA | 442 / 500 | 313 / 129 | 35.1% [29.9, 40.3] | 8.5% [3.9, 13.4] |
| QERA | 442 / 500 | 313 / 129 | 24.0% [19.6, 28.9] | 7.8% [3.3, 13.0] |

## BIRD, scored snapshot, Luna verifier

Luna verifier on the qwen3.8-flash scored snapshot. Tasks used and planned come from that snapshot's analysis, not a completed 500-task collect.

| Run | Solver model | Solver effort | Verifier model | Verifier effort | Jev model | Temperature |
|---|---|---|---|---|---|---:|
| `verifier-luna/bird-scienceworld-scored` | `qwen3.8-flash` | Provider default | `gpt-6-luna` | Provider default | `typesafe-ai/jev` | 0 |

### Train

| View | Tasks used / planned | Native correct / incorrect | False rejection (95% CI) | False acceptance (95% CI) |
|---|---:|---:|---:|---:|
| QA | 464 / 500 | 358 / 106 | 40.2% [35.2, 45.5] | 61.3% [52.1, 70.5] |
| QEA | 464 / 500 | 358 / 106 | 19.3% [15.5, 23.7] | 70.8% [61.9, 79.2] |
| QRA | 464 / 500 | 358 / 106 | 42.5% [37.4, 47.7] | 50.9% [41.2, 60.2] |
| QERA | 464 / 500 | 358 / 106 | 31.0% [26.7, 35.8] | 55.7% [46.7, 64.7] |

### Test

| View | Tasks used / planned | Native correct / incorrect | False rejection (95% CI) | False acceptance (95% CI) |
|---|---:|---:|---:|---:|
| QA | 370 / 500 | 270 / 100 | 36.7% [31.2, 42.5] | 45.0% [35.5, 54.6] |
| QEA | 370 / 500 | 270 / 100 | 14.8% [10.6, 19.3] | 52.0% [42.1, 61.9] |
| QRA | 370 / 500 | 270 / 100 | 49.6% [43.6, 55.7] | 32.0% [23.1, 41.4] |
| QERA | 370 / 500 | 270 / 100 | 47.0% [40.8, 52.7] | 28.0% [19.2, 36.8] |

## ScienceWorld, scored snapshot, Luna verifier

Luna verifier on the qwen3.8-flash scored snapshot. Tasks used and planned come from that snapshot's analysis, not a completed 500-task collect.

| Run | Solver model | Solver effort | Verifier model | Verifier effort | Jev model | Temperature |
|---|---|---|---|---|---|---:|
| `verifier-luna/bird-scienceworld-scored` | `qwen3.8-flash` | Provider default | `gpt-6-luna` | Provider default | `typesafe-ai/jev` | 0 |

### Train

| View | Tasks used / planned | Native correct / incorrect | False rejection (95% CI) | False acceptance (95% CI) |
|---|---:|---:|---:|---:|
| QA | 213 / 500 | 163 / 50 | 69.3% [62.2, 75.6] | 4.0% [0.0, 10.9] |
| QEA | 213 / 500 | 163 / 50 | 42.9% [35.7, 50.7] | 8.0% [1.7, 16.4] |
| QRA | 213 / 500 | 163 / 50 | 90.2% [85.2, 94.5] | 10.0% [2.2, 18.6] |
| QERA | 213 / 500 | 163 / 50 | 43.6% [35.9, 51.5] | 10.0% [2.1, 18.9] |

### Test

| View | Tasks used / planned | Native correct / incorrect | False rejection (95% CI) | False acceptance (95% CI) |
|---|---:|---:|---:|---:|
| QA | 81 / 500 | 73 / 8 | 61.6% [50.6, 72.6] | 12.5% [0.0, 42.9] |
| QEA | 81 / 500 | 73 / 8 | 39.7% [28.6, 51.4] | 0.0% [0.0, 0.0] |
| QRA | 81 / 500 | 73 / 8 | 93.2% [87.3, 98.6] | 12.5% [0.0, 42.9] |
| QERA | 81 / 500 | 73 / 8 | 37.0% [26.0, 48.6] | 25.0% [0.0, 60.0] |

[Full-precision metrics](summary.csv) · [Paired comparisons](comparisons.csv)
