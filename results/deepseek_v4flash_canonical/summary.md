# deepseek_v4flash_canonical — run summary

Models: `deepseek-v4-flash` · Seeds: [np.int64(0), np.int64(1), np.int64(2)] · Tasks: 38 · Trajectories: 204

## Pass@1 by system

| system | passed | total | Pass@1 | mean wallclock |
|---|---:|---:|---:|---:|
| deepseek_baseline | 75 | 103 | 72.82% | 610s |
| deepseek_omicverse | 88 | 101 | 87.13% | 675s |

## Pass@1 by layer × system

| layer | deepseek_baseline | deepseek_omicverse |
|---|---:|---:|
| A | 15/15 (100%) | 15/15 (100%) |
| B | 21/30 (70%) | 26/30 (87%) |
| C | 5/12 (42%) | 8/12 (67%) |
| E | 13/18 (72%) | 17/17 (100%) |
| F | 8/8 (100%) | 8/8 (100%) |
| G | 9/10 (90%) | 10/10 (100%) |
| L | 4/10 (40%) | 4/9 (44%) |

## Failure-mode breakdown (failed trajectories only)

| system             |   exceeded_turns |   wrong_tool_choice |   wrong_workflow_order |
|:-------------------|-----------------:|--------------------:|-----------------------:|
| deepseek_baseline  |                6 |                  14 |                      8 |
| deepseek_omicverse |                2 |                   7 |                      4 |

