# minimax_full — run summary

Models: `MiniMax-M2.7` · Seeds: [np.int64(0), np.int64(1), np.int64(2)] · Tasks: 38 · Trajectories: 228

## Pass@1 by system

| system | passed | total | Pass@1 | mean wallclock |
|---|---:|---:|---:|---:|
| minimax_baseline | 88 | 114 | 77.19% | 902s |
| minimax_omicverse | 91 | 114 | 79.82% | 908s |

## Pass@1 by layer × system

| layer | minimax_baseline | minimax_omicverse |
|---|---:|---:|
| A | 14/15 (93%) | 15/15 (100%) |
| B | 18/30 (60%) | 23/30 (77%) |
| C | 6/12 (50%) | 7/12 (58%) |
| E | 14/18 (78%) | 15/18 (83%) |
| F | 12/12 (100%) | 12/12 (100%) |
| G | 14/15 (93%) | 13/15 (87%) |
| L | 10/12 (83%) | 6/12 (50%) |

## Failure-mode breakdown (failed trajectories only)

| system            |   exceeded_turns |   wrong_tool_choice |   wrong_workflow_order |
|:------------------|-----------------:|--------------------:|-----------------------:|
| minimax_baseline  |                3 |                  13 |                     10 |
| minimax_omicverse |                3 |                  13 |                      7 |

