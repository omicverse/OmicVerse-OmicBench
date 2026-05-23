# deepseek_v4_pro_full — run summary

Models: `deepseek-v4-pro` · Seeds: [np.int64(0), np.int64(1), np.int64(2)] · Tasks: 38 · Trajectories: 228

## Pass@1 by system

| system | passed | total | Pass@1 | mean wallclock |
|---|---:|---:|---:|---:|
| deepseek_baseline | 81 | 114 | 71.05% | 922s |
| deepseek_omicverse | 102 | 114 | 89.47% | 811s |

## Pass@1 by layer × system

| layer | deepseek_baseline | deepseek_omicverse |
|---|---:|---:|
| A | 15/15 (100%) | 13/15 (87%) |
| B | 18/30 (60%) | 26/30 (87%) |
| C | 4/12 (33%) | 8/12 (67%) |
| E | 14/18 (78%) | 16/18 (89%) |
| F | 10/12 (83%) | 12/12 (100%) |
| G | 12/15 (80%) | 15/15 (100%) |
| L | 8/12 (67%) | 12/12 (100%) |

## Failure-mode breakdown (failed trajectories only)

| system             |   exceeded_turns |   wrong_tool_choice |   wrong_workflow_order |
|:-------------------|-----------------:|--------------------:|-----------------------:|
| deepseek_baseline  |                8 |                  15 |                     10 |
| deepseek_omicverse |                1 |                   8 |                      3 |

