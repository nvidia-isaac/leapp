# Automatic segmentation supports only a linear segment chain (v1)

When a marked region interleaves torch and warp, LEAPP auto-splits it into single-kind nodes
at each bridge. In v1 this is supported **only when the segments form a linear chain** — each
segment's bridged output feeds the next segment and nothing else. If a tensor forks across a
bridge (e.g. a torch tensor `h` is both sent into a warp segment via `wp.from_torch` *and*
still consumed by a parallel torch branch in the same region), LEAPP fails loudly at trace
time and directs the user to express the fork as explicit manually-named nodes.

## Why

A forked tensor would have to be live in two node contexts at once, which LEAPP's tracer
already forbids (`validate_status` rejects mixing active node contexts inside a traced
function). Supporting intra-region DAGs would require a materially more complex segmentation
pass. We lose no expressiveness by deferring it: arbitrary DAGs are already expressible
*across* manually-named nodes, which LEAPP wires by tag-matching. So v1 keeps the auto-split
simple and predictable, and the manual-node path covers everything else.

## Consequences

A future reader who forks a tensor across a bridge inside one region will hit an explicit
error rather than a silently wrong graph. Lifting this to full intra-region DAG support later
is additive and does not change the linear-chain behavior.
