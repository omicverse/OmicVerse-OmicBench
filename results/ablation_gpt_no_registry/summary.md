# ablation_gpt_no_registry — run summary

Models: `gpt-5.5` · Seeds: [np.int64(0)] · Tasks: 38 · Trajectories: 38

## Pass@1 by system

| system | passed | total | Pass@1 | mean wallclock |
|---|---:|---:|---:|---:|
| gpt_omicverse_no_registry | 24 | 38 | 63.16% | 337s |

## Pass@1 by layer × system

| layer | gpt_omicverse_no_registry |
|---|---:|
| A | 5/5 (100%) |
| B | 5/10 (50%) |
| C | 2/4 (50%) |
| E | 3/6 (50%) |
| F | 3/4 (75%) |
| G | 5/5 (100%) |
| L | 1/4 (25%) |

## Failure-mode breakdown (failed trajectories only)

| system                      |   wrong_tool_choice |   wrong_workflow_order |
|:----------------------------|--------------------:|-----------------------:|
| gpt_omicverse_no_registry |                   9 |                      5 |

