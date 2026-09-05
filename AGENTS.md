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
- Write distributions as `\text{Normal}`, `\text{Binomial}`, `\text{HalfNormal}`, never `\mathcal{N}`. Use `\mathbb{E}` for expectations, `\perp` for independence and `\mid` for conditioning.
- Add short remarks for the natural variants (for example, what changes if the outcome is continuous).
- For the expected value, use the letter $\text{E}$ in math mode.
- For the probability, use the letter $\text{P}$ in math mode.


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
