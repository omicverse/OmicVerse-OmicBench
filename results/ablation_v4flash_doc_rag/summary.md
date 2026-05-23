# ablation_v4flash_doc_rag — run summary

Models: `deepseek-v4-flash` · Seeds: [np.int64(0)] · Tasks: 38 · Trajectories: 38

## Pass@1 by system

| system | passed | total | Pass@1 | mean wallclock |
|---|---:|---:|---:|---:|
| deepseek_omicverse_doc_rag | 27 | 38 | 71.05% | 800s |

## Pass@1 by layer × system

| layer | deepseek_omicverse_doc_rag |
|---|---:|
| A | 5/5 (100%) |
| B | 8/10 (80%) |
| C | 3/4 (75%) |
| E | 3/6 (50%) |
| F | 4/4 (100%) |
| G | 2/5 (40%) |
| L | 2/4 (50%) |

## Failure-mode breakdown (failed trajectories only)

| system                     |   wrong_tool_choice |   wrong_workflow_order |
|:---------------------------|--------------------:|-----------------------:|
| deepseek_omicverse_doc_rag |                   9 |                      2 |

