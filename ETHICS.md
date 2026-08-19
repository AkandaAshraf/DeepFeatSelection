# What this work is for

This statement is not a licence condition. The code is Apache-2.0 and the
text CC BY 4.0, both deliberately permissive, because the methods here are
most useful to hospitals, universities and companies whose legal teams cannot
accept use-restricted licences — the very people best placed to apply them.
Restricting use would have expressed an intention at the cost of achieving
it.

So this is a request rather than a requirement, and it is the reason the work
exists.

**The purpose.** These tools were built to help people make better decisions
in research that ultimately reaches patients: which variables in a large
recorded system are driven by the rest, and how much an observational
similarity really licenses a claim about intervention. The work is personal,
unfunded, and unconnected to any employment. It is published so that others
can use it, check it, and correct it.

**What using it well looks like.**

- Report the controls with the results. Every scan carries its own gate
  report; a number quoted without it is not this method's output.
- Respect the stated limits. MACE is blind to sources by design and has high
  precision with low recall: absence from a result is not evidence of
  autonomy. The redundancy calibration measures cell lines, not tumours, and
  nominates no clinical action.
- Publish what fails. The negative results in this repository are load-bearing.
  If this method gives you a null, that null is worth the same as a hit.
- Do not use it to manufacture confidence you do not have — in a target, a
  diagnosis, or a decision affecting a person — by omitting the controls that
  would have qualified it.

**What would be a misuse.** Presenting these statistics as evidence of
clinical benefit; using them to justify a decision about an individual
patient; stripping the controls to make a finding look stronger than it is;
or using the work in surveillance or targeting of people. None of these are
forbidden by the licence. They are simply contrary to why any of it was
written.

If you build on this, the most valuable thing you can send back is the case
where it fails.
