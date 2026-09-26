# Demystifying the Silence of Correctness Bugs in the PyTorch Compiler

This repository is the artifact for our study on silent correctness bugs in the
PyTorch compiler (`torch.compile`). It contains the labeled bug dataset from our
empirical study (RQ1), the detection benchmark and the results of five existing
DL-compiler testing techniques (RQ3), the results of our guideline-driven agent
(RQ4), and the extended-window data used in the discussion.

## Directory layout

```
.
├── README.md
├── discussion
│   └── extended_issues_with_labels.csv
├── rq1
│   ├── final_228_with_labels.csv
│   └── total_7034_issues.csv
├── rq3
│   ├── benchmark_issues.csv
│   └── bugs_detected_by_selected_techniques.csv
└── rq4
    ├── bugs_detected_by_agents.csv
    ├── bug_reports/
    └── total_new_bugs.csv
```

## RQ1 — Bug characteristic study

- `rq1/total_7034_issues.csv` — every `torch.compile` issue we collected and ran
  through our filtering funnel (7,034 issues). Columns: `issue title`, `link`,
  `filtering type` (the funnel step at which the issue was kept or discarded;
  `Studied Dataset (Confirmed Correctness Bug)` marks the 228 studied bugs).
- `rq1/final_228_with_labels.csv` — the 228 confirmed correctness bugs of the
  core study window (up to 2026-01-31) with their manual labels. Columns:
  `title`, `link`, `root cause`, `bug-triggering pattern`.

## RQ3 — Detection benchmark and existing techniques

- `rq3/benchmark_issues.csv` — the 130 qualified (load-ready) benchmark bugs.
  Columns: `issue title`, `issue link`, `pull request link`, `fixing commit id`,
  `root cause`, `bug-triggering pattern`.
- `rq3/bugs_detected_by_selected_techniques.csv` — the 31 benchmark bugs
  detected by at least one of the five evaluated techniques (TitanFuzz, Opera,
  NNSmith, DeepConstr, WhiteFox, NeuRI). Columns: `title`, `issue link`,
  `root cause`, `bug-triggering pattern`, `Detected by Techniques` (semicolon
  separated list of the techniques that detected the bug).

## RQ4 — Guideline-driven agent

- `rq4/bugs_detected_by_agents.csv` — the distinct bugs found by `cgagent`
  (ReAct agent with our guidelines) and `baseagent` (ReAct agent without
  guidelines) across three rounds.
  Columns: `bug_identifier`, `tool_name`, `bug-triggering pattern`.
- `rq4/total_new_bugs.csv` — all previously unknown bugs discovered by `cgagent`
  and reported to PyTorch. Columns: `name`, `status`, `link`, `high priority`,
  `bug-triggering pattern`.
- `rq4/bug_reports/` — the reproducible code for each bug identifier found across
  the three rounds.

## Discussion — Extended study window

- `discussion/extended_issues_with_labels.csv` — the additional confirmed bugs
  from the extended window (2026-02-01 to 2026-04-30), labeled with the same
  taxonomy as RQ1. Columns: `title`, `link`, `root cause`,
  `bug-triggering pattern`.
