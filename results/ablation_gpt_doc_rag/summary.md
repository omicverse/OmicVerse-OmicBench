# ablation_gpt_doc_rag — run summary

Models: `gpt-5.5` · Seeds: [np.int64(0)] · Tasks: 38 · Trajectories: 38

## Pass@1 by system

| system | passed | total | Pass@1 | mean wallclock |
|---|---:|---:|---:|---:|
| gpt_omicverse_doc_rag | 29 | 38 | 76.32% | 466s |

## Pass@1 by layer × system

| layer | gpt_omicverse_doc_rag |
|---|---:|
| A | 5/5 (100%) |
| B | 6/10 (60%) |
| C | 4/4 (100%) |
| E | 3/6 (50%) |
| F | 4/4 (100%) |
| G | 5/5 (100%) |
| L | 2/4 (50%) |

## Failure-mode breakdown (failed trajectories only)

| system                  |   wrong_tool_choice |   wrong_workflow_order |
|:------------------------|--------------------:|-----------------------:|
| gpt_omicverse_doc_rag |                   5 |                      4 |

