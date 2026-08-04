# IJF submission checklist: Open-Source Forecasting special section

Working checklist for submitting `ijof.tex` to the International Journal of Forecasting special section on open-source forecasting. Sources: the call for papers (forecasters.org) and the IJF/Elsevier guide for authors. Items marked done reflect the state of the manuscript on this branch.

## Logistics

- [ ] Submit via https://submit.elsevier.com/IJF selecting article type "SI: Open-Source Forecasting".
- [ ] Submission window: opens 2026-03-03 (open now), deadline 2026-08-31.
- [x] Cover letter (explicitly required by the call): drafted in `cover_letter.tex` (compiled to `cover_letter.pdf`); summarizes the contribution, the fit to the special section scope, and transparently flags the self-citation blinding caveat.
- Guest editors: Mitchell O'Hara-Wild (Monash), Anastasios Panagiotelis (Monash), Tim Januschowski (Databricks).

## Double-anonymized review (IJF requirement, currently NOT satisfied)

- [x] Comment out the author block in `ijof.tex` (authors, affiliations, `\ead`, `\cortext`) exactly as `IJFTemplate/ijftemplate.tex` shows; the submitted manuscript body must contain no identifying information. Done; `ijof.pdf` regenerated and verified (page 1 shows no authors or affiliations).
- [x] Prepare a separate title page file with: full author names, affiliations including department, city, and country, corresponding author email, acknowledgments, and all declarations. Done in `title_page.tex` (compiled to `title_page.pdf`); `[TODO]` placeholders remain for coauthor emails, affiliation city/country details, and ORCID.
- [x] Move the Acknowledgments section out of the manuscript body onto the title page (it names the authors' communities and collaborators). Done; the section is commented out in `ijof.tex` for restoration after acceptance.
- [x] Decide how to handle the `orduz_*` self-citation keys plus first-person phrasing ("our worked example"). Decision: keep as-is; the cover letter transparently explains that the paper surveys the authors' published worked examples, that anonymizing the self-citations would dismantle the paper's structure, and defers to the guest editors on any further blinding.

## Declarations to prepare (title page / submission system)

- [x] Declaration of competing interest: drafted on the title page (notes employment at PyMC Labs, Google DeepMind, Amazon); confirm wording before submitting.
- [x] CRediT author-contribution statement for all three authors: drafted on the title page; confirm role assignment with coauthors.
- [x] Funding statement: "no external funding" wording on the title page.
- [x] Data availability statement: on the title page (public or simulated data; FreshRetailNet-50K on Hugging Face; code in public notebooks).
- [x] Generative-AI disclosure per Elsevier policy: standard statement included on the title page (AI used for readability and language; authors take full responsibility).
- [ ] ORCID for the corresponding author (encouraged): `[TODO]` placeholder on the title page.

## Manuscript items

- [x] Abstract within the 100-150 word limit (trimmed to 149 words; code notation removed per the no-jargon rule).
- [x] At least five keywords (six present).
- [x] elsarticle options `[11pt,3p,review,authoryear]` match the official IJF template; `model5-names` bibliography style.
- [ ] Coauthor emails: `[TODO]` placeholders on the title page (Du Phan, Theo Rashid); collect and fill in before uploading.
- [x] Optional highlights file (3-5 bullets): decided to skip; "where applicable" per the guide, not mandatory.

## Reproducibility hardening (recommended, not required)

- [ ] Consider archiving the case-study notebooks as a DOI'd snapshot (e.g. Zenodo) and citing the archive; blog URLs can rot and the call emphasizes transparency and reproducibility.
- Note on the scaling claim: the paper's "approximately ten minutes on a single GPU via Modal" quotes the cited blog post; the published notebook documents the 5.5-minute model fit but not the Modal orchestration itself. Keep the Modal run reproducible (script or notes) in case a referee asks.

## Referee-risk notes (no action required, be ready in the response letter)

- The call lists "benchmarking, forecast evaluation" as a topic; the paper deliberately reports no systematic benchmarks (Section 3.4 states this and Section 7 frames the missing benchmark suite as an open problem). Be ready to defend the qualitative-scaling stance or add one systematic comparison if asked.
- The call names R, Python, and Julia; the paper is Python/JAX-only. A one-sentence scoping acknowledgment could preempt the question.
- "Mature libraries" in the call invites maturity evidence (tests, releases, adoption) for NumPyro and the auxiliary packages; consider one sentence if revising.
- Consider foregrounding (blinding permitting) that a coauthor is a NumPyro core author; it strengthens the "we speak for this software" position.
