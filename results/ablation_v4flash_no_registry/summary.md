# ablation_v4flash_no_registry — run summary

Models: `deepseek-v4-flash` · Seeds: [np.int64(0)] · Tasks: 38 · Trajectories: 38

## Pass@1 by system

| system | passed | total | Pass@1 | mean wallclock |
|---|---:|---:|---:|---:|
| deepseek_omicverse_no_registry | 32 | 38 | 84.21% | 745s |

## Pass@1 by layer × system

| layer | deepseek_omicverse_no_registry |
|---|---:|
| A | 5/5 (100%) |
| B | 8/10 (80%) |
| C | 4/4 (100%) |
| E | 4/6 (67%) |
| F | 4/4 (100%) |
| G | 4/5 (80%) |
| L | 3/4 (75%) |

## Failure-mode breakdown (failed trajectories only)

| system                         |   wrong_tool_choice |   wrong_workflow_order |
|:-------------------------------|--------------------:|-----------------------:|
| deepseek_omicverse_no_registry |                   4 |                      2 |

