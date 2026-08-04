# Commit-hash provenance across the history rewrite

On 2026-08-04 an `.env` file — committed on 2025-12-14 and carrying a Roboflow
API key — was purged from the whole history with `git filter-repo`. The key
itself was rotated first; the purge is hygiene, not remediation, because a blob
that has been public cannot be un-published.

The rewrite changed **every commit hash from 2025-12-14 onward**. Repository
*content* is unaffected: the tree hash of the final commit is byte-identical
before and after (`6fddfbc3c7d2d0964007ec6f74acee7ac1bc8d06`).

## Why this file exists

Every run under `results/` records the `git_sha` that produced it, in its
`run_meta.json` — 269 of them. Those SHAs refer to the pre-rewrite history and
no longer resolve. That would silently break the link between a reported number
and the code that produced it, which is the one thing this repository's
reproducibility story rests on.

`commit-map-2026-08-04-env-purge.tsv` is the mapping. Two columns, tab-separated,
`old` → `new`, 176 entries. To resolve a `git_sha` recorded in any pre-rewrite
run:

```bash
grep ^<old-sha> docs/provenance/commit-map-2026-08-04-env-purge.tsv
```

Two commits map to `0000000000000000000000000000000000000000`: they became empty
once `.env` was removed (their only content was adding or deleting that file)
and were pruned. Neither carried any other change.

## Rollback

A complete pre-rewrite bundle of every ref was taken before the operation and
verified with `git bundle verify`. It is deliberately **not** in the repository —
it contains the leaked blob. It lives outside the working tree; ask the
repository owner if a pre-rewrite hash ever needs resolving beyond this map.

## A gap this map does not close

24 `run_meta.json` entries reference 10 distinct SHAs that are absent from the
map. Those commits **did not exist in this repository before the purge either** —
verified by `git cat-file -t` against the pre-rewrite object store. They are
casualties of an *earlier* history rewrite, not of this one. Nothing here can
recover them; they are recorded so that a reader who finds an unresolvable
`git_sha` knows which of the two rewrites to blame, and that this one is
accounted for.
