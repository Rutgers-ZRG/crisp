# Public Release Notes

This branch is intended to contain only reusable source, documentation, tests,
and sanitized examples.

Excluded from the public branch:

- Local benchmark scripts with private paths.
- Scheduler logs and benchmark outputs.
- Generated structures and checkpoint archives.
- Model weights and potential files.
- Local editor, cache, and agent state.

Before publishing a remote repository, run:

```bash
git status --short --branch
python -m unittest discover -s tests
git grep -nE 'TO[K]EN|SE[C]RET|PASS[W]ORD|PRIVATE[ ]KEY'
```
