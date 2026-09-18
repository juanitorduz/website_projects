# Personal Website Projects

## Development

Activate the Python environment:

```bash
pixi shell
```

## No em-dashes

Do not use em-dashes (`—`) in any prose. Use the most natural alternative for the grammatical role the dash was playing: a colon for an explanation or expansion, a comma (or pair of commas) for a parenthetical aside, parentheses for a softer aside, a semicolon for a closely-related independent clause, or a full stop to start a new sentence. Pick the form that reads most cleanly; do not just substitute one punctuation mark mechanically for another.

## No hard line breaks in prose

When writing text files (`.txt`, `.md`, `.qmd`, and similar), do **not** wrap prose at a fixed column. Write each paragraph as a single long line and let the editor/renderer handle visual wrapping. This also applies to Jupytext notebooks and scripts (`.py`).

- Yes: one line per paragraph, one line per bullet.
- No: inserting newlines every 80 (or 100, or any other) characters inside a paragraph.

Exceptions: code blocks, tables, YAML front matter, and anything where the newline is semantically meaningful (e.g. markdown lists, mermaid diagrams): keep those formatted normally.

## American English spelling

Use American English spelling. Do not use British English spelling.

## Writing Style

Use ASD-STE100 simplified technical English as the baseline: short sentences, active voice, one idea per sentence, the same word for the same thing.

### Prose

- Write in the voice of the existing posts: succinct, precise, not flamboyant. Use "we". No hype, no jokes, no metaphors. Model sentence: "PathMC is a package from PyMC Labs that removes that boilerplate."
- Open a notebook with the motivation in words, then what the notebook does, then what it assumes of the reader. State the problem abstractly and say what we want to estimate and how before any math or code.
- Section headers are `## Title Case` without numbers. The usual skeleton is: intro, `## Prepare Notebook`, conceptual sections, data, model, fit and diagnostics, results, `## Conclusion`, `## References`.
- References are real and linked inline (see "No fake citations"). Verify every URL before publishing.

### Math

- Define every symbol in prose before it appears in a display formula (what is $n$, what is $N_j$). Introduce the abstract estimand before the estimator.
- Write distributions as `\text{Normal}`, `\text{Binomial}`, `\text{HalfNormal}`, never `\mathcal{N}`. Use `\text{E}` for expectations, `\text{P}` for probabilities, `\perp` for independence and `\mid` for conditioning.
- Add short remarks for the natural variants (for example, what changes if the outcome is continuous).
- For the expected value, use the letter $\text{E}$ in math mode.
- For the probability, use the letter $\text{P}$ in math mode.

### Voice

- Write as the analyst walking the reader through the work: "we" for what the notebook does, "let's" for transitions, "you" when giving advice. Do not write impersonally or in the passive voice.
- Every code cell gets a one-sentence lead-in that says what we are about to do ("Next, let's look at the two SEM series over time.", "The same scores in tabular form:"). Every plot or table gets one or two sentences right after it that say what we see.
- Open paragraphs with plain signposts: "First", "Next", "Now", "Finally", "Observe that", "Note that", "Recall", "In other words", "Here is an important observation:", "Here are two important remarks:", "Keep this in mind."
- It is fine to introduce a step with a question and then answer it ("How should you read the mediator?", "What would we have planned without the funnel?"). A few times per notebook, not in every paragraph.
- Say plainly whether a check passed: "which is good!", "The media parameters are stable!", "The two builds agree!", "quite reasonable", "not surprising", "remarkable". At most one exclamation mark per section, on a passed check or a genuine surprise. One or two emojis per notebook (🚀, 😎, 🙌, 😅) are fine at a celebratory or self-deprecating moment.

### Sentences and paragraphs

- One idea per paragraph, one to three sentences, about 30 words on average. Split anything longer.
- Prefer full stops. Do not use semicolons. Use a colon only to introduce a list or a single explanation.
- Every sentence has a subject and a verb. Do not open a paragraph with a verbless noun phrase and a colon. Not "The decision-relevant question: ...", "Same bar as the first iteration: ...", "Three planning choices, each stated so they can be challenged:". Write "Here we are interested in the following question:", "We hold the model to the same bar as the first iteration:", "We make three planning choices here, and you should feel free to challenge each of them:".
- Do not use bold run-in pseudo-headings ("**Why this parametrization.** The spend term is linear because ..."). Use a real heading, an admonition with a bold title line, or a plain sentence.
- No meta-commentary about the notebook or the writing ("They are boring, so we spell them out", "a result this notebook did not script", "Read the table for what it is", "measured rather than asserted"). State the observation.
- Use modest claim verbs: "check", "confirm", "see", "support". Not "prove", "certify", "entitled to".
- Emphasis: italics for a single word (*position*, *meaning*, *true*); bold for the one key term of a sentence (**before** fitting, the **direct** path). Do not bold whole clauses.

### Formatting

- Every number in prose goes in math mode: `$26.9\%$`, `$209$` weeks, `$0.67$`, `$94\%$` HDI, `$33K$`. Never a bare `26.9%`. Dates stay plain (`2018-08-05`).
- Every channel, column, variable, parameter, and attribute name goes in backticks: `Direct Mail`, `sem_spend`, `funnel_lambda`, `mmm.channel_data_scaled`. Never bare "TV, Radio, and Newspaper".
- Cross-reference other notebooks with `{ref}` and the phrase "see {ref}`mmm_example` for more details". Reference API objects with `{class}`, `{meth}`, `{func}`.
- Ground modeling decisions in a linked paper or blog post. Quote the relevant passage in a `>` block when it carries the argument.

### Structure

- Headings are short Title Case noun phrases (one to four words): "The SEM Funnel", "Model Fitting", "Budget Optimization", "Guardrail Sensitivity". No questions, no "vs", no colon subtitles at H2. Reuse the standard MMM section names and order: Prepare Notebook, Load Data, Exploratory Data Analysis, Model Specification, Prior Predictive Checks, Model Fitting, Model Diagnostics, Posterior Predictive Checks, Media Deep Dive (Channel Contributions, Saturation Curves, Return on Ad Spend (ROAS)), Budget Optimization, Conclusion, Next Steps.
- The introduction states the business problem, then a numbered outline whose items match the H2 headings, then a `{note}` pointing to the notebooks that cover the building blocks.
- List items open with a bold lead phrase: `1. **Budget anchor.** We anchor ...`, `- **Additive channel effects**: ...`.
- The Conclusion has `### Key Findings` (bullets with bold leads), `### Model Limitations` (bullets), `### Recommendations` (numbered), and then a `## Next Steps` numbered list with bold leads. Not a sequence of bold run-in paragraphs.


### Causal DAGs

Draw DAGs with graphviz, never as markdown or ASCII art. Build `dag = gr.Digraph()` (with `import graphviz as gr`), add nodes and edges, and leave `dag` as the last expression of the cell without `;`.

- Nodes are `style="filled"` with this palette: blue `#2a2eec80` for the treatment, exposure or covariates of interest; green `#328c0680` for the outcome; orange `#fa7c1780` for a secondary node of interest (mediator, collider, conditioned or selection node); `lightgray` for unobserved variables.
- Draw a conditioned node as a box with `shape="box"`. Put the role in the label on a second line, for example `label="U\n(unobserved)"` or `label="I = 1\n(responded)"`.
- Color the arrows that break an assumption red (`color="red"`). Use `dag.subgraph(name="cluster_...")` with a `label` to show two DAGs side by side.

```python
dag = gr.Digraph()
dag.node("X", label="X\n(covariates)", color="#2a2eec80", style="filled")
dag.node("Y", label="Y\n(outcome)", color="#328c0680", style="filled")
dag.node("I", label="I = 1\n(responded)", shape="box", color="#fa7c1780", style="filled")
dag.edge("X", "I")
dag.edge("X", "Y")
dag
```

### Plots

Use arviz >= 1 for statistical plots and plain matplotlib for the rest. Every notebook starts with this setup:

```python
seed: int = 42
rng: np.random.Generator = np.random.default_rng(seed=seed)

HDI_PROB = 0.94

az.style.use("arviz-darkgrid")
az.rcParams["stats.ci_kind"] = "hdi"
az.rcParams["stats.ci_prob"] = HDI_PROB
plt.rcParams["figure.figsize"] = [10, 6]
plt.rcParams["figure.dpi"] = 100
plt.rcParams["figure.facecolor"] = "white"
```

- Titles are bold and size 18: `fig.suptitle("...", fontsize=18, fontweight="bold")`. For an arviz plot collection use `pc.viz["figure"].item().suptitle(...)`.
- Prefer the arviz plot functions (`az.plot_dist`, `az.plot_forest`, `az.plot_trace_dist`, `az.plot_prior_posterior`) over hand-made matplotlib versions. Add reference values with `az.add_lines(pc, values={...}, visuals={"ref_line": {...}})`. Compare models by passing a dict `{"name": idata, ...}` and call `pc.add_legend("model")`.
- Use the default matplotlib color cycle (`C0`, `C1`, ...). Add a legend whenever a plot has more than one series. End every plot cell with `;` (see "Keep ; after the end of a plot").

### Tooling

- Use polars for tables, never pandas.
- Generate synthetic data with a PyMC generative model: define the model, fix the true parameters with `pm.do`, and take one draw with `pm.sample_prior_predictive(draws=1, random_seed=rng)`. Do not simulate with raw numpy.
- Use `pymc.dims` (`import pymc.dims as pmd`) where it fits; where the dims module lacks a distribution, convert with `.values` and use the classic API for that line.
- Pass the `rng` generator to `random_seed=` in every sampling call.

## No fake citations

Do not use fake citations. If you need to cite a source, ensure it exists.

## Keep ; after the end of a plot

In jupyter notebooks and Jupytext scripts, keep the `;` after the end of a plot.
