# Open items

Ordered by the project's rule: studies first, dissemination when there is
something finished to disseminate.

## Studies (first)

- [x] DepMap calibration phase B results -> read against the four
      pre-registered predictions (paper/depmap_protocol.md)
- [x] DepMap bootstrap CIs (B=100 pre-registered; B=1000 precision
      check confirms them)
- [x] TNBC arm - run and reported as underpowered, as declared; the
      definition failed its own positive control first and was replaced
- [x] `r_obs(A|rest)` many-to-one arm - answered in the negative
      (AUC 0.614 panel vs 0.607 best-pair)
- [ ] Mutant phenotyping on IMMOBILISED worm datasets (freely-moving line
      closed with a negative result; see ledger 2026-08-16)
- [ ] Literature items still needing the author's eyes: Kato 2015 Fig. 5B
      (does AVB appear, what does its bar show); Randi/Leifer 2023 functional
      atlas AVA->AVB entry; whether RIM is in Uzel 2022's hub set; Ray &
      Gordus 2025 (AIB driven by AVA)
- [ ] Decide whether to cite the 2016 tourism/greenhouse-gas causality paper
      as prior work in any v2

## Dissemination (after results, priority order)

- [ ] Bluesky + comp-neuro mailing list announcement once the paper DOI is
      live — identified as the fastest route to the audience that actually
      uses whole-brain data
- [ ] LinkedIn post (drafted; waiting on the DOI so the first link unfurls)
- [ ] DepMap community forum post when the calibration is written up — the
      room where the practitioners who rely on these matrices sit
- [ ] Glioma-network collaboration letter (Winkler/Monje-line labs): the
      driven-follower map as the complement to their pacemaker finding

## Infrastructure / legacy

- [ ] ORCID: create, link to arXiv + Zenodo, claim the four name spellings
- [ ] Zenodo DOI per release of the repository
- [x] Licence decided (2026-08-17): code Apache-2.0, text CC BY 4.0, intent
      in a non-binding ETHICS.md. Ethical-source licences (Hippocratic 3.0,
      Do No Harm) were rejected because they are not OSI-open and the
      institutions most likely to apply this clinically cannot accept
      use-restricted terms — the restriction would have cost the goal it
      expressed. Apache over MIT for the explicit patent grant.
- [x] pip-installable `mace` package with a five-line worked example — the
      compounding discoverability item (agreed: should exist)
- [ ] one-page HTML how-to for the author's personal site, linking package,
      paper and repo; fuller documentation later
- [ ] Manuscript v2 batch: byline spacing, drop "draft" from the date line,
      remaining literature items above

## Added 2026-08-20

- [x] iEEG same-scale sensitivity arms: DONE, and the line is closed.
      Laplacian and bipolar-skip were declared in advance, both cleared the
      validity gate (rho 0.64 / 0.59 vs raw's 0.01) and both corroborated
      P-S1 in the discovery cohort. A pre-registered replication on 10
      held-out subjects then FAILED on all five predictions, with the
      primary arm in the wrong direction. Per the declaration, no further
      cohort is sought and the discovery result is withdrawn. See
      FINDINGS.md 8b and the ledger.
- [ ] OPTIONAL, only if it can be pre-registered on unexamined data: the two
      iEEG cohorts differ by recording site (NIH/PY/rns vs jh/pt/umf). A
      site-heterogeneity study would have to be declared as its own
      hypothesis on data not yet looked at. Running it on the existing
      cohorts would be the post-hoc subgroup hunt already declined.
- [ ] PLOS: check submission status; confirm the Publication Fee Assistance
      application attached at initial submission (it cannot be added later).
- [ ] Verify by eye the citations sourced from search rather than read:
      Thompson 2021 author list, Pacini 2024 review details, De Kegel 2026
      preprint.
- [ ] LinkedIn article: update to the corrected numbers (ceiling 17.0%
      [9.7, 27.3], lift 39x, lineage removal 56% at r2>=0.6) and add the
      De Kegel/Ryan positioning before posting.
- [ ] Colab notebook: fill in the DepMap download URLs and run it end to end
      once, to confirm it works from a cold start.

## Added 2026-08-22 (after the boundary-map / bottleneck run)

DONE, recorded in the ledger and protocols:
- [x] Boundary map on known ground truth (51 cells) - two frontiers found,
      one method defect claimed and then refuted by its own test
- [x] Threshold rule comparison - MAX stands, our defect claim refuted
- [x] Duplicate channels - source blindness shown CONDITIONAL ON SATURATION,
      up to 50% of sources flagged off ceiling
- [x] Gate statistic comparison - no ghost statistic protects; saturation
      must be reported directly
- [x] TF/PyTorch fidelity - Spearman 0.955, the two pipelines are
      commensurable
- [x] Perturb-seq calibration - richer phenotype gives a LOWER ceiling
      (4.0% vs DepMap's 17.0%); no correction to the submission needed
- [x] Bottleneck width and binary-vs-float codes - the detection cap is a
      property of b=32, not of the method; b of order 2V needed; capacity
      not geometry
- [x] Source-detection retry at capacity - the confound WAS capacity and is
      fixed; rejection stands only because the signal is too weak

OPEN, generated by the above:
- [ ] HIGH VALUE. Does outflow strengthen with coupling? The retry left this
      unsettled: sources sit only 0.002 above the ghost at coupling 0.3. If a
      harder-driving source gives proportionally more outflow, the source
      detector is alive; if not, the line is dead. Cheap, and it decides the
      biggest open problem in the portfolio.
- [ ] Confirm b~2V on a DIFFERENT system class before it becomes deployment
      guidance. One generating process is not enough for a rule that would
      change every future scan.
- [ ] Saturation as a formal gate criterion, pre-registered on a held-out
      generating process. Declined deliberately on the motivating data.
- [ ] Report self-baseline saturation alongside every published scan - it is
      already computed and has never been shown. Needs no experiment.
- [ ] Wider binary codes (b=256, 512): the grid could not separate whether
      binary needs more width or hits a growing constant factor.
- [ ] Boundary map's redundancy row is mislabelled - it measures driver
      enrichment, not target redundancy. Relabel in the published table.
- [ ] OPTIONAL, needs a SYMMETRIC declaration: re-run BOTH iEEG discovery and
      replication at b~2V. Capacity attenuates rather than reverses, so it is
      unlikely to explain the failure, and re-running only the disappointing
      arm is forbidden.
