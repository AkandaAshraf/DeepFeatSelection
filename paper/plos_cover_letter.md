# Cover letter — PLOS Computational Biology

Dear Editors,

Please consider the enclosed manuscript, *"If two genes look interchangeable,
are they? A calibration of expression redundancy against interventional
equivalence across 43 million gene pairs"*, for publication in PLOS
Computational Biology.

The paper measures a quantity that computational biology relies on
constantly and that, as far as I could establish, has not been published.
When two genes track each other closely in expression data, they are
routinely treated as a single hypothesis: one is followed up on the
understanding that the other would have behaved the same. That step is a
conditional probability, and it can be measured directly on DepMap, where
expression profiles and genome-wide CRISPR knockout screens exist for the
same 1,103 cell lines. Scoring all 43.9 million gene pairs gives the answer:
observational redundancy raises the odds of interventional equivalence about
39-fold (95% CI 23–65), and the ceiling is 17.0% (95% CI 9.7–27.3). Among
the 13 most redundant pairs in the dataset, none were equivalent.

I would draw your attention to three features that I believe fit the
journal's aims.

**The adjacent literature is engaged precisely rather than displaced.** The
paralog-buffering work of De Kegel and Ryan asks whether losing one paralog
increases dependency on the other. That is a question about compensation,
measured on paralogs; interventional equivalence is a different relation
measured over the whole gene universe, and the two point in opposite
directions. The manuscript recovers that predicted opposition in its own
data and reports it as convergence with their result rather than as a novel
finding.

**Two pre-registered controls failed, and both failures are reported in
full.** One null control was, by simple algebra, incapable of differing from
the analysis it was meant to test; it was replaced with one that can fail.
The positive control then failed on the declared similarity measure, which
correctly indicted the measure rather than the biology: correlation across
cell lines discards the shared mean in which uniform co-essentiality lives.
Both the failing and the corrected measures are reported. A declared
subgroup analysis is reported as underpowered rather than rescued.

**The work is fully reproducible.** The pre-registration was written before
any curve was computed, all data are public DepMap releases downloaded by
the analysis scripts, and every number in the manuscript is re-derived from
its output file by an audit script. Code, pre-registration and the complete
experiment ledger — including voided results — are archived at
doi:10.5281/zenodo.21988145.

This work is personal, unfunded, and unconnected to any employment. I am an
independent researcher and have no institutional support for publication
costs; I would be grateful for consideration under the PLOS Publication Fee
Assistance programme.

The manuscript is not under consideration elsewhere, and all data and code
necessary to reproduce it are public. I have no competing interests to
declare.

Thank you for your consideration.

Yours sincerely,

Akanda Wahid-Ul-Ashraf
Independent Researcher
akandaashraf@outlook.com
