# qwen_local_full — run summary

Models: `qwen3.6:35b-a3b-256k` · Seeds: [np.int64(0)] · Tasks: 38 · Trajectories: 76

## Pass@1 by system

| system | passed | total | Pass@1 | mean wallclock |
|---|---:|---:|---:|---:|
| mini_swe_baseline | 17 | 38 | 44.74% | 517s |
| mini_swe_omicverse | 30 | 38 | 78.95% | 418s |

## Pass@1 by layer × system

| layer | mini_swe_baseline | mini_swe_omicverse |
|---|---:|---:|
| A | 3/5 (60%) | 5/5 (100%) |
| B | 4/10 (40%) | 8/10 (80%) |
| C | 1/4 (25%) | 2/4 (50%) |
| E | 2/6 (33%) | 4/6 (67%) |
| F | 3/4 (75%) | 4/4 (100%) |
| G | 4/5 (80%) | 5/5 (100%) |
| L | 0/4 (0%) | 2/4 (50%) |

## Failure-mode breakdown (failed trajectories only)

| system             |   exceeded_turns |   silent_none |   wrong_tool_choice |   wrong_workflow_order |
|:-------------------|-----------------:|--------------:|--------------------:|-----------------------:|
| mini_swe_baseline  |               15 |             2 |                   1 |                      3 |
| mini_swe_omicverse |                5 |             3 |                   0 |                      0 |

