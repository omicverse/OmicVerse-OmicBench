# glm_full — run summary

Models: `glm-5.1` · Seeds: [np.int64(0), np.int64(1), np.int64(2)] · Tasks: 38 · Trajectories: 224

## Pass@1 by system

| system | passed | total | Pass@1 | mean wallclock |
|---|---:|---:|---:|---:|
| glm_baseline | 76 | 112 | 67.86% | 713s |
| glm_omicverse | 98 | 112 | 87.50% | 858s |

## Pass@1 by layer × system

| layer | glm_baseline | glm_omicverse |
|---|---:|---:|
| A | 15/15 (100%) | 15/15 (100%) |
| B | 18/30 (60%) | 24/30 (80%) |
| C | 6/12 (50%) | 11/12 (92%) |
| E | 11/18 (61%) | 15/18 (83%) |
| F | 10/12 (83%) | 11/12 (92%) |
| G | 15/15 (100%) | 14/15 (93%) |
| L | 1/10 (10%) | 8/10 (80%) |

## Failure-mode breakdown (failed trajectories only)

| system        |   exceeded_turns |   wrong_tool_choice |   wrong_workflow_order |
|:--------------|-----------------:|--------------------:|-----------------------:|
| glm_baseline  |                8 |                  17 |                     11 |
| glm_omicverse |                2 |                   6 |                      6 |

