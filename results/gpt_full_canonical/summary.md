# gpt_full_canonical — run summary

Models: `gpt-5.5` · Seeds: [np.int64(0), np.int64(1), np.int64(2)] · Tasks: 38 · Trajectories: 227

## Pass@1 by system

| system | passed | total | Pass@1 | mean wallclock |
|---|---:|---:|---:|---:|
| gpt_baseline | 82 | 114 | 71.93% | 241s |
| gpt_omicverse | 103 | 113 | 91.15% | 351s |

## Pass@1 by layer × system

| layer | gpt_baseline | gpt_omicverse |
|---|---:|---:|
| A | 15/15 (100%) | 14/15 (93%) |
| B | 20/30 (67%) | 27/30 (90%) |
| C | 5/12 (42%) | 9/12 (75%) |
| E | 10/18 (56%) | 18/18 (100%) |
| F | 12/12 (100%) | 12/12 (100%) |
| G | 15/15 (100%) | 15/15 (100%) |
| L | 5/12 (42%) | 8/11 (73%) |

## Failure-mode breakdown (failed trajectories only)

| system          |   wrong_tool_choice |   wrong_workflow_order |
|:----------------|--------------------:|-----------------------:|
| gpt_baseline  |                  22 |                     10 |
| gpt_omicverse |                   7 |                      3 |

