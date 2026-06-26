## 2026-06-26 Final Review Fixes

- Fixed `leapp.compile_graph(visualize=True)` to propagate visualization failures instead of logging and continuing, so missing SVG/PNG artifacts now fail the compile step.
- Added a regression test that patches `LeappGraph.visualize()` to raise and asserts `compile_graph()` raises the same error.
- Updated the docs homepage export bundle example to list `visualization.svg` as the primary graph artifact with `visualization.png` alongside it.
- Removed `.superpowers/sdd/task-5-report.md` and `.superpowers/sdd/task-6-report.md` from the Git index while keeping the local files in place.
