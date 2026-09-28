# Working agreement

Start from main, use the workstream branches listed in branches.json, and
merge changes through a focused pull request. These are starting branches,
not separate permanent copies of shared code. Keep them current by merging
main; avoid rebasing or force-pushing someone else's work.

Run `python3 scripts/check.py` before publishing. CI checks scaffolding and
metadata; it does not certify a solver, trained model, or physical result.
A reference port additionally needs manufactured/analytic checks, independent
RHS and step comparisons, and grid/time/domain refinement on declared cases.

Record dependency commit hashes and configuration changes in experiment
manifests. Change shared contracts in core first, then update downstream pins.
Never commit credentials, full cluster home paths, raw fields, or checkpoints.
This scaffold does not choose a software license for future research code.
Agree on a license with collaborators before describing the project as open source.
