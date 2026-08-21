# What is in this repository's history

The current tree is authoritative. Git history, as in any repository, also
contains superseded versions. This file says plainly what those are, so that
nothing has to be discovered.

History is **not** rewritten, deliberately. The experiment ledger cites
commit hashes as evidence of when declarations were made — for example that
the iEEG replication predictions were fixed at `93a95be` before any held-out
recording was downloaded, and that the label-handling rules were fixed at
`67571af` before any label was opened. Those citations are what make the
pre-registrations checkable by a stranger. Rewriting history would change
every hash and break that chain, replacing verifiable timestamps with
"the references were updated to match". For a project whose credibility
rests on checkable declarations, that trade is not worth making.

## Superseded material reachable from history

**Manuscript LaTeX sources** (`paper/excess_paper.tex`, `paper/main.tex`,
`paper/refs.bib`, `paper/depmap_paper.tex`, figure sources). These were
later untracked as a matter of preference — the compiled preprint is the
artefact meant to be read — not because anything in them was wrong.

**Submission drafts** (`paper/depmap_plos.pdf`, `paper/plos_cover_letter.md`,
`paper/depmap_paper.pdf`). Working documents for journal submission, kept out
of the public record by choice.

**A manuscript version with two incorrect citations.** An early PLOS draft
attributed two real papers to the wrong authors: the Cancer Dependency Map
review to Pacini et al. rather than Arafeh et al., and the paralog review to
Parvin et al. rather than Ryan et al. Both were found by verification after
submission, corrected in the current tree, and reported to the journal. The
uncorrected version remains in history.

## Findings that were published here and later changed

These are recorded in full in [the ledger](paper/causal_detection_log.md),
which keeps withdrawn results with a pointer to whatever overturned them
rather than deleting them.

- **Two DepMap numbers were wrong when first published here.** The bootstrap
  confidence intervals were computed from one histogram bin while claiming
  two, making them too wide; and the lineage-inflation figure was quoted at a
  lower redundancy threshold than the surrounding text implied. Both were
  found by re-deriving every quoted value from its output file, and both are
  corrected.
- **The iEEG SOZ result was recorded as corroborated and is now withdrawn.**
  It passed in a 28-subject cohort and survived two post-hoc sensitivity
  arms, then failed a pre-registered replication on 10 held-out subjects on
  all five predictions. See FINDINGS.md §8b.

## The rule this follows

A result that does not survive scrutiny is marked as such and kept, with the
evidence that overturned it. Nothing is quietly removed.
