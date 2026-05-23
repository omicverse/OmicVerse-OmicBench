# gemini_full — run summary

Models: `gemini-3.1-flash-lite-preview` · Seeds: [np.int64(0), np.int64(1), np.int64(2)] · Tasks: 38 · Trajectories: 224

## Pass@1 by system

| system | passed | total | Pass@1 | mean wallclock |
|---|---:|---:|---:|---:|
| gemini_baseline | 71 | 112 | 63.39% | 333s |
| gemini_omicverse | 89 | 112 | 79.46% | 445s |

## Pass@1 by layer × system

| layer | gemini_baseline | gemini_omicverse |
|---|---:|---:|
| A | 12/15 (80%) | 15/15 (100%) |
| B | 19/30 (63%) | 24/30 (80%) |
| C | 3/12 (25%) | 5/12 (42%) |
| E | 9/18 (50%) | 15/18 (83%) |
| F | 12/12 (100%) | 12/12 (100%) |
| G | 12/15 (80%) | 13/15 (87%) |
| L | 4/10 (40%) | 5/10 (50%) |

## Failure-mode breakdown (failed trajectories only)

| system           |   exceeded_turns |   silent_none |   wrong_tool_choice |   wrong_workflow_order |
|:-----------------|-----------------:|--------------:|--------------------:|-----------------------:|
| gemini_baseline  |                4 |             2 |                  25 |                     10 |
| gemini_omicverse |                5 |             2 |                  12 |                      4 |

