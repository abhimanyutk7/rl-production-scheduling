# Sources, reuse and project contributions

This file connects implemented elements to their sources. The report contains
the full bibliography and measured comparisons. An established method, a cited
idea and imported software are different kinds of reuse.

## Source-to-implementation map

| Element | Source | Relationship to this project |
|---|---|---|
| Processing-duration generator and seeds | [Taillard (1993), OR-Library job-shop generator](https://people.brunel.ac.uk/~mastjjb/jeb/orlib/jobshopinfo.html) | `factory_rl/instances.py` reproduces the generator. The project samples/scales durations and generates its extended routes, arrivals and deadlines. |
| PPO | [Schulman et al. (2017)](https://arxiv.org/abs/1707.06347) | Established policy optimisation objective, executed through SB3-Contrib. |
| Invalid-action masking | [Huang and Ontanon (2020)](https://arxiv.org/abs/2006.14171) | Existing method for restricting action distributions to permitted choices. Project masks encode factory and service constraints. |
| RL implementation | [Stable-Baselines3, Raffin et al. (2021)](https://jmlr.org/papers/v22/20-1364.html) and [SB3-Contrib MaskablePPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html) | Imported software supplies PPO rollout collection, advantage calculation, losses and parameter updates. Those library internals are not claimed as project-authored algorithms. |
| Shared learned dispatch motivation | [Zhang et al. (2020)](https://arxiv.org/abs/2010.12367) | Conceptual motivation for learned job scoring. Our shared MLP differs from the paper's graph-based model. |
| Cross-entropy policy search | [Szita and Lorincz (2006)](https://doi.org/10.1162/neco.2006.18.12.2936) | Established episodic policy-search concept adapted to readable factory priorities and service thresholds. This is not a reproduction of their Tetris system. |
| Residual learning | [Johannink et al. (2018 preprint)](https://arxiv.org/abs/1812.03201) | General motivation for corrections to an existing controller. Our bounded job-logit correction is a factory adaptation, not the paper's robot controller. |
| Grouped transitions | [Sutton, Precup and Singh (1999)](https://doi.org/10.1016/S0004-3702(99)00052-1) | Temporal-abstraction motivation. Grouping changes the GAE time index; matching trajectories does not establish identical optimisation. |
| GAE | [Schulman et al. (2015 preprint)](https://arxiv.org/abs/1506.02438) | Existing advantage estimator provided through the learning library. |
| Particle condition estimates | [Gordon, Salmond and Smith (1993)](https://doi.org/10.1049/ip-f-2.1993.0015) | Established sequential Monte Carlo idea. Factory wear, measurement and survival models are project assumptions. |
| Inspection and maintenance | [Koutas, Bismut and Straub (2024)](https://doi.org/10.1186/s43065-024-00098-9) | Related research; the paper uses different learning/inference approaches. Its results are not our results. |
| Actor/critic pretraining | [Kernbach et al. (2026)](https://arxiv.org/abs/2602.23804) | Motivation for a separately controlled refinement experiment. Our evidence did not establish an isolated critic-pretraining gain. |
| RL scheduling hyperheuristics | [Lassoued et al. (2026)](https://arxiv.org/abs/2601.11189) | Literature comparison. Our actor ranks jobs rather than implementing their complete rule-switching system. |

## Software and templates

`pyproject.toml` declares the imported numerical, simulation, optimisation and
learning libraries. PyTorch supplies neural-network operations; Gymnasium
supplies the environment interface; OR-Tools supplies the CP-SAT solver used by
the original reference baseline. Standard dependency code remains attributed to
its upstream authors. A package import is not a new algorithm contribution.

The source ZIP includes the user-supplied IEEE Access template and IEEEtran class
with their notices. The local class adaptation addresses XeTeX colour support
and academic-project metadata, as described in the ZIP's README. The report is
an unpublished course project, not an IEEE publication.

The final PPT follows the user-supplied qubit-routing deck's visual format.
That deck is a design reference, not a source of factory experiments. The
user-supplied performance handout guides evaluation coverage, not measured data.

## Implemented contribution and evidence

The implemented work integrates the extended factory simulator, public
information boundary, condition estimates, persistent billing, service/dispatch
wrappers, shared scorer, readable search and controlled evaluation protocols.
This does not establish that the individual components, or every combination of
scheduling, maintenance and energy, are previously unknown.

The [consolidated report](outputs/inspection_research/report.md) preserves both
positive and negative comparisons. The
[locked selection](outputs/inspection_research/final_selection/selection.json)
identifies saved policies before the final test. The
[review script](scripts/review_submission.py) checks the saved neural checkpoint
and final-test grid and can execute a short CPU replay.

These citations document provenance. They are not a plagiarism-checker result
or a certification of exclusive originality.
