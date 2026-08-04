# IJF submission checklist: Open-Source Forecasting special section

Working checklist for submitting `ijof.tex` to the International Journal of Forecasting special section on open-source forecasting. Sources: the call for papers (forecasters.org) and the IJF/Elsevier guide for authors. Items marked done reflect the state of the manuscript on this branch.

## Logistics

- [ ] Submit via https://submit.elsevier.com/IJF selecting article type "SI: Open-Source Forecasting".
- [ ] Submission window: opens 2026-03-03 (open now), deadline 2026-08-31.
- [ ] Cover letter (explicitly required by the call): draft one summarizing the contribution and its fit to the special section scope.
- Guest editors: Mitchell O'Hara-Wild (Monash), Anastasios Panagiotelis (Monash), Tim Januschowski (Databricks).

## Double-anonymized review (IJF requirement, currently NOT satisfied)

- [ ] Comment out the author block in `ijof.tex` (authors, affiliations, `\ead`, `\cortext`) exactly as `IJFTemplate/ijftemplate.tex` shows; the submitted manuscript body must contain no identifying information.
- [ ] Prepare a separate title page file with: full author names, affiliations including department, city, and country (current `\address` entries are bare organization names), corresponding author email, acknowledgments, and all declarations.
- [ ] Move the Acknowledgments section out of the manuscript body onto the title page (it names the authors' communities and collaborators).
- [ ] Decide how to handle the 16 `orduz_*` self-citation keys (25 in-text occurrences) plus first-person phrasing ("our worked example"). The guide's convention is citing as [Anonymous, year] with "details omitted for double-blind review" in the reference list. Full anonymization is structurally awkward for this paper (the case studies are the authors' published blog posts), so consider asking the guest editors how they want survey-of-own-work papers handled before mechanically anonymizing.

## Declarations to prepare (title page / submission system)

- [ ] Declaration of competing interest (note author affiliations: PyMC-Labs, Google, Amazon).
- [ ] CRediT author-contribution statement for all three authors.
- [ ] Funding statement (state "no external funding" if applicable).
- [ ] Data availability statement: all case studies use public or simulated data; FreshRetailNet-50K is on Hugging Face (Dingdong-Inc/FreshRetailNet-50K); code is in public notebooks.
- [ ] Generative-AI disclosure per Elsevier policy: required if AI tools were used in the writing process; decide and declare.
- [ ] ORCID for the corresponding author (encouraged).

## Manuscript items

- [x] Abstract within the 100-150 word limit (trimmed to 149 words; code notation removed per the no-jargon rule).
- [x] At least five keywords (six present).
- [x] elsarticle options `[11pt,3p,review,authoryear]` match the official IJF template; `model5-names` bibliography style.
- [ ] Coauthor emails: two TODO placeholders remain in `ijof.tex` (Du Phan, Theo Rashid); collect and fill in on the title page.
- [ ] Optional highlights file (3-5 bullets); "where applicable" per the guide, not mandatory.

## Reproducibility hardening (recommended, not required)

- [ ] Consider archiving the case-study notebooks as a DOI'd snapshot (e.g. Zenodo) and citing the archive; blog URLs can rot and the call emphasizes transparency and reproducibility.
- Note on the scaling claim: the paper's "approximately ten minutes on a single GPU via Modal" quotes the cited blog post; the published notebook documents the 5.5-minute model fit but not the Modal orchestration itself. Keep the Modal run reproducible (script or notes) in case a referee asks.

## Referee-risk notes (no action required, be ready in the response letter)

- The call lists "benchmarking, forecast evaluation" as a topic; the paper deliberately reports no systematic benchmarks (Section 3.4 states this and Section 7 frames the missing benchmark suite as an open problem). Be ready to defend the qualitative-scaling stance or add one systematic comparison if asked.
- The call names R, Python, and Julia; the paper is Python/JAX-only. A one-sentence scoping acknowledgment could preempt the question.
- "Mature libraries" in the call invites maturity evidence (tests, releases, adoption) for NumPyro and the auxiliary packages; consider one sentence if revising.
- Consider foregrounding (blinding permitting) that a coauthor is a NumPyro core author; it strengthens the "we speak for this software" position.
