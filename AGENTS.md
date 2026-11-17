# Personal Website Projects

## Development

Activate the Python environment:

```bash
pixi shell
```

## Notebook Workflow

- Edit a notebook through its Jupytext percent script (`.py`), convert it back with `jupytext --update` so the cell ids stay the same, execute the notebook end to end, and run `pre-commit`. Commit the `.ipynb`.
- Read every number in the prose from the executed outputs, and re-read the prose after every rerun. Every claim about the results must be visible in a figure, a table or an `assert`. A claim from theory needs a derivation or a citation instead.
- Treat a question in a review as a question. Check it, and push back with the reason when the suggestion does not hold. Put the answer in the notebook only when the reader needs it.

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
- The introduction is one connected story. Start with a concrete example of the problem in plain words, say why it is not trivial, and cite the key references (as a bullet list when there are several). Then say what the notebook does, what it keeps simple on purpose (with a link to more), and what it assumes of the reader. State the problem abstractly and say what we want to estimate and how before any math or code.
- State the assumptions and the scope as a bullet list with bold leads (`- **No lead time:** An order for a day arrives at the start of that day.`). Do not compress them into one sentence such as "We leave X, Y and Z out of scope."
- Use the words of the application domain and the simplest common term ("next month" for a retailer, not "next quarter"). Explain a technical phrase the first time it appears, or replace it ("each week is a newsvendor problem", "within a divergence ball", "the path under the true future").
- Introduce a concept before the text uses it. Do not mention a term, a solver step or a later quantity before the section that defines it. Move the sentence there, or point to that section by its name in italics.
- When a later section uses a metric or a check again, recall its definition and why it matters in one sentence.
- Justify every claim that is not obvious: show why in one or two steps, or cite the source of the result (for example the newsvendor critical fractile or the Rockafellar and Uryasev representation). When a check fails or a figure shows a pattern (a poor coverage, an upward trend), explain why instead of only reporting it. Interpret the sign of a parameter in the terms of the domain.
- Make a concept concrete with a small worked example or a plot before we use it on the data ("Let's work a simple example.").
- Before a non-trivial function, explain its idea and the intuition in the markdown cell above the code.
- Do not justify a step by what the code needs ("since the plots below use them", "because the generator of the next section needs a `Priors` tuple"). Motivate it by what it shows the reader.
- Do not add asides the reader does not need. Name a variant that we do not study in one sentence at most ("It is easy to adapt the techniques below to the backlogging case."), and do not work out how it differs. Do not compare a step with another library or example in the main text ("This is the NumPyro counterpart of `pm.do`"). Do not mention unmerged pull requests, drafts or unreleased features.
- Section headers are `## Title Case` without numbers. The usual skeleton is: intro, `## Prepare Notebook`, conceptual sections, data, model, fit and diagnostics, results, `## Conclusion`, `## References`.
- References are real and linked inline (see "No fake citations"). Verify every URL before publishing.
- Link earlier blog posts for background (`jax.lax.scan`, an introduction to SVI) with "For more details, see ..." instead of explaining the topic again.
- A follow-up post opens with a link to the earlier post and a short summary of it. Recall each reused term or result in one plain sentence at its first use ("Recall from the first post that ..."), so that a reader who skipped the earlier post can follow. Do not repeat its derivations or figures.

### Math

- Define every symbol in prose before it appears in a display formula (what is $n$, what is $N_j$). Introduce the abstract estimand before the estimator.
- Do not use one letter for two things (`x` for the order and for the features, `S` for the number of scenarios and for the sales $S_t$). Use the usual notation of the field ($a$ for an action in decision theory) and compare it with the notation of the key reference. The indices of a symbol show how it varies (a capacity that differs by store is $C_s$ everywhere).
- Write the data generating process of every model as a LaTeX block. Connect each formula to the code that implements it, for example which function or dictionary holds the transform.
- Put integrals and multi-step derivations in display math (`$$ ... $$` or `\begin{align*}`), one formula per line. Put a sum in display math when it is the result of a step. Do not write an integral inline. Do not leave a blank line inside an `align*` block, because the rendered page then shows a literal `$$`.
- Write distributions as `\text{Normal}`, `\text{Binomial}`, `\text{HalfNormal}`, never `\mathcal{N}`. Use `\text{E}` for expectations, `\text{P}` for probabilities, `\perp` for independence and `\mid` for conditioning.
- Add short remarks for the natural variants (for example, what changes if the outcome is continuous).
- For the expected value, use the letter $\text{E}$ in math mode.
- For the probability, use the letter $\text{P}$ in math mode.

### Voice

- Write as the analyst walking the reader through the work: "we" for what the notebook does, "let's" for transitions, "you" when giving advice. Do not write impersonally or in the passive voice.
- Every code cell gets a one-sentence lead-in that says what we are about to do and why ("Next, let's look at the two SEM series over time.", "We now fix the true parameter values that we intervene on, and take one draw."). The setup cell under `## Prepare Notebook` needs none.
- Do not narrate the plotting call or a standard encoding ("with `az.plot_forest`, one row per plan with the mean and ...", "We pack the draws into a small tree"). Do not restate settings that the code shows ("four chains of $1{,}000$ draws each"). Explain a code idea that the reader needs (what `jax.vmap` batches), and say how to read a figure whose encoding is not standard.
- Every plot or table gets one or two sentences or a short bullet list right after it that say what we see. Give the takeaway (a trend, a pattern, a consequence), not the values that the reader can read from the figure. Do not make up a takeaway.
- Open paragraphs with plain signposts: "First", "Next", "Now", "Finally", "Observe that", "Note that", "Recall", "In other words", "Here is an important observation:", "Here are two important remarks:", "Keep this in mind."
- Open a paragraph with the context that leads to the step: "Having understood the intuition of the CVaR, we can write the objective that the CVaR plan minimizes:". Do not open with a bare statement that has no link to the text before it ("We also need the tail average.", "We ask for a service level instead of a cost.").
- It is fine to introduce a step with a full question and then answer it ("How should you read the mediator?", "What would we have planned without the funnel?"). A few times per notebook, not in every paragraph. Do not open a paragraph with a short question fragment ("Why the margin?", "Why price?").
- End a section with a transition to the next one ("Now that we have the covariates, we can describe the demand model."). Do not end a section on a code detail ("The `Problem` tuple travels through every function below").
- Say plainly whether a check passed: "which is good!", "The media parameters are stable!", "The two builds agree!", "quite reasonable", "not surprising", "remarkable". At most one exclamation mark per section, on a passed check or a genuine surprise. One or two emojis per notebook (🚀, 😎, 🙌, 😅) are fine at a celebratory or self-deprecating moment.

### Sentences and paragraphs

- One idea per paragraph, one to three sentences, about 30 words on average. Split anything longer. Keep the sentences of one point in one paragraph, and do not split a short reading into one-sentence paragraphs.
- Do not pack many numbers or facts into one sentence. Split such a sentence into short sentences or bullets. Start with the motivation and the intuition, say what each term means, and stop when the point is clear.
- When a paragraph lists several items (costs, components, metrics, results by plan), write them as bullets, nested if that helps. Write steps in order as a numbered list.
- Prefer full stops. Do not use semicolons. Use a colon only to introduce a list or a single explanation. Do not stage a reveal with a contrast and a colon ("The buffer sits in the stock available, not in every order: ..."). Do not write aphorisms ("we keep that loop and change the object it acts on").
- Every sentence has a subject and a verb. Do not open a paragraph with a verbless noun phrase and a colon. Not "The decision-relevant question: ...", "Same bar as the first iteration: ...", "Three planning choices, each stated so they can be challenged:". Write "Here we are interested in the following question:", "We hold the model to the same bar as the first iteration:", "We make three planning choices here, and you should feel free to challenge each of them:".
- Do not use bold run-in pseudo-headings ("**Why this parametrization.** The spend term is linear because ..."). Use a real heading or a plain sentence. Bold leads in list items are fine (see *Structure*). In a paragraph, the one exception is a remark for a side note, such as a caveat for real applications or an implementation detail: `**Remark:**` or `**Remark (Continuous Orders).**`.
- No meta-commentary about the notebook or the writing ("They are boring, so we spell them out", "a result this notebook did not script", "Read the table for what it is", "measured rather than asserted"). State the observation. Do not repeat a stock sentence ("Here is what the figure shows.") through the notebook.
- Use modest claim verbs: "check", "confirm", "see", "support". Not "prove", "certify", "entitled to". Write "consistent with" for a cause that no output tests.
- Use the precise statistical term instead of a loose word: "right-skewed" or "a point mass at zero", not "a lump at zero".
- Emphasis: italics for a single word (*position*, *meaning*, *true*). Bold a new term where we define it ("Let $H$ be the **number of days in the planning horizon** and let $D_t$ be the **demand on day** $t$."). Elsewhere, bold at most the one key term of a sentence (**before** fitting, the **direct** path). Do not bold whole clauses.
- Refer to a section of the notebook by its name in italics: "We check this in the *Out of Sample Evaluation* section."

### Formatting

- Every number in prose goes in math mode: `$26.9\%$`, `$209$` weeks, `$0.67$`, `$94\%$` HDI, `$33K$`. Never a bare `26.9%`. Dates stay plain (`2018-08-05`).
- Show the arithmetic behind a derived count in parentheses, for example where the number of latent variables of a model comes from.
- Every channel, column, variable, parameter, and attribute name goes in backticks: `Direct Mail`, `sem_spend`, `funnel_lambda`, `mmm.channel_data_scaled`. Never bare "TV, Radio, and Newspaper".
- Ground modeling decisions in a linked paper or blog post. Quote the relevant passage in a `>` block when it carries the argument.

### Structure

- Headings are short Title Case noun phrases (one to four words): "The SEM Funnel", "Model Fitting", "Budget Optimization", "Guardrail Sensitivity". No questions, no "vs", no colon subtitles at H2. Reuse the standard MMM section names and order: Prepare Notebook, Load Data, Exploratory Data Analysis, Model Specification, Prior Predictive Checks, Model Fitting, Model Diagnostics, Posterior Predictive Checks, Media Deep Dive (Channel Contributions, Saturation Curves, Return on Ad Spend (ROAS)), Budget Optimization, Conclusion, Next Steps.
- Split a long section into `###` (and `####`) subsections, for example *Model Specification*, *Data Preparation* and *Scenario Visualization*. A section that introduces a concept (a Pareto frontier) starts with what it is and why we care.
- The introduction states the business problem, then a numbered outline whose items match the H2 headings.
- List items open with a bold lead phrase: `1. **Budget anchor.** We anchor ...`, `- **Additive channel effects**: ...`.
- Keep the compared items (plans, models, metrics) in one order everywhere: the order in which we introduce them, which is also the order of their `StrEnum`, their colors, every legend and every bullet list. Show every compared item in each comparison, not a subset.
- Remove a section, a figure or a paragraph that adds no information to the story, including a figure without a takeaway. Shorten a section whose result we expect a priori to its main figure.
- The Conclusion has `### Key Findings` (bullets with bold leads), `### Model Limitations` (bullets), `### Recommendations` (numbered), and then a `## Next Steps` numbered list with bold leads. Not a sequence of bold run-in paragraphs. Key Findings are general lessons that could carry over to other data sets, with noun-phrase leads. They do not summarize the numbers of this notebook.
- Technical cross-checks (for example a custom solver against a convex solver) go in an `## Appendix` after `## Next Steps` and before `## References`. The main text points to it.

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

- Set the interval kind once in the setup cell and never per plot. A notebook that reads band edges as percentiles, or that plots a quantity with a point mass, uses `az.rcParams["stats.ci_kind"] = "eti"` with the constant `CI_PROB`. Label every band with its kind (`94% ETI`) and define the interval once in the text.
- Titles are bold and size 18: `fig.suptitle("...", fontsize=18, fontweight="bold")`. For an arviz plot collection use `pc.viz["figure"].item().suptitle(...)`. Do not put a count of draws or series in a title ("30 Joint Demand Scenarios"). A diagnostic summary, such as the in-sample CRPS of a posterior predictive check, can go in the title through an f-string.
- Prefer the arviz plot functions (`az.plot_dist`, `az.plot_forest`, `az.plot_trace_dist`, `az.plot_prior_posterior`) over hand-made matplotlib versions such as `ax.hist`. Add reference values with `az.add_lines(pc, values={...}, visuals={"ref_line": {...}})`. Compare models by passing a dict `{"name": idata, ...}` and call `pc.add_legend("model")`.
- Draw bands over time with `az.plot_lm`, never with `ax.fill_between`, and show the $50\%$ band together with the $94\%$ band.
- Show results as figures, not as printed values or tables: a bar plot for a few scores, a line plot for a sweep, a forest plot for intervals. Do not use `print`. A short preview of the data (`df.head()`, a `group_by(...).len()` count) is fine. Round the numbers of a displayed table and keep it short.
- Use seaborn for bar plots (`sns.barplot` with `hue_order`, `palette` and `errorbar=None`) and `sns.heatmap` for a calendar or a grid. Do not draw error bars on a bar plot. Show the distribution (`az.plot_dist`) or a forest plot instead.
- Use a diagram for a structure that is hard to follow in prose, for example a partition of the products into roles.
- Draw a time series as a line, not as points. This includes the observed data and the true future.
- Stack panels that share the x axis but not the y axis with `nrows=2, ncols=1, sharex=True`. Use `col_wrap=3` when a grid gets too wide. Give a figure its own cell when it needs its own reading.
- Put the legend below the axes, outside the plot area: `fig.legend(loc="outside lower center", ncol=...)`. A figure does not rely on the legend of another figure. Every color and reference line in a legend has a label, and the legend has no title. Build the legend from labeled artists (`label=...` or `ax.get_legend_handles_labels()`). Add a hand-made `Patch` only for an artist that cannot carry a label, such as the cells of a heatmap.
- Keep the default font sizes of legends, labels and ticks. Never make text smaller. Keep axis labels short enough to fit the axis.
- Use the default matplotlib color cycle (`C0`, `C1`, ...). Add a legend whenever a plot has more than one series. End every plot cell with `;` (see "Keep ; after the end of a plot").
- Give each entity (a plan, a model) one color in the whole notebook, kept in a dict such as `PLAN_COLORS`, and do not use that color for another meaning. Draw a mean in the color of its draws, thicker and opaque. Do not use gray for data, because the `arviz-darkgrid` background is gray.
- Reference lines of one kind in a figure (the true values, the bounds) share one style. Keep the default line width unless a line needs emphasis, such as a mean drawn over its draws. Check that no reference line is drawn twice.
- Format percentage axes with `PercentFormatter(xmax=1, decimals=0)`, or `decimals=1` when the values are a few percent, and count axes with integer ticks (`MaxNLocator(integer=True)`). Put numeric values on a numeric axis, not on evenly spaced categories.
- Add bar labels only when the values matter, as plain numbers without a box (`ax.bar_label(bars, fmt="{:,.0f}")`).
- Render every NumPyro model with `numpyro.render_model(..., render_distributions=True)`. When a model is built from components, render each component first and then the whole model, with a bold title on each graph.

### Tooling

- Use polars for tables, never pandas.
- Generate synthetic data with a PyMC or NumPyro generative model: define the model, fix the true parameters with `pm.do` (NumPyro: `numpyro.handlers.do`), and take one draw with `pm.sample_prior_predictive(draws=1, random_seed=rng)` (NumPyro: `Predictive`). Do not simulate with raw numpy.
- Use `pymc.dims` (`import pymc.dims as pmd`) where it fits; where the dims module lacks a distribution, convert with `.values` and use the classic API for that line.
- Pass the `rng` generator to `random_seed=` in every sampling call. A JAX notebook replaces `rng` with `rng_key: PRNGKeyArray = random.PRNGKey(seed)` and derives every key with `random.split`.

### Code

- Keep constants and settings in `NamedTuple`s with descriptive names (`Setup`, `SolverConfig`, not `Problem`). Split a large tuple into one tuple per component. Compute a value that follows from another field (the number of groups is the length of `group_cross`) in a property or a method, and do not store it twice.
- Use a `NamedTuple` for a function that returns several values and for a collection of results (plans, fits), a `StrEnum` for labels (metrics, roles, models), and `Final` constants for site names.
- Define every function, constant and variable in the cell of its first use.
- Write small functions with descriptive names. Replace a dense condition with a named helper, and reuse a helper instead of repeating its logic. When the callers of a function differ in one step (the order rule of a plan), pass that step as a function typed with a `Protocol`. Do not pass an argument that some callers ignore.
- Use named arrays or dimensions instead of positional slices of a packed array (`covariates[:, :-n]`).
- Leave a blank line before and after an `if`, `for` or `with` block.
- In a JAX notebook, stay in JAX: `jnp` instead of `np`, `jax.scipy.stats` instead of `scipy.stats`, `jax.grad` instead of a hand-derived gradient, and the NumPyro transforms instead of a hand-written inverse or Jacobian. Avoid `np.asarray` conversions.
- Type the functions with `jaxtyping` and check them at run time with `beartype` (`%load_ext jaxtyping`, `%jaxtyping.typechecker beartype.beartype`). Use the library types (`ScalarLike`) instead of custom aliases.
- Keep the library defaults (`chain_method`, the sampler settings) unless the text gives a reason. Remove conversions, arguments and helpers that do nothing (an `astype`, a `float(...)`, a `reparam` flag whose value is fixed, a metric that the notebook does not use).
- Check a claim about the code with an `assert` in the notebook (a projection is idempotent, a plan meets its constraint).

### Bayesian Workflow

- Elicit the priors on the scale of the data with [PreliZ](https://preliz.readthedocs.io/) and justify each one with a plot of its density and of the implied quantity on the data scale. Put what we know in the prior: a price elasticity is centered on a negative value.
- Run a prior predictive check before the fit and a posterior predictive check after it.
- With synthetic data, check the parameter recovery: plot each posterior with its true value as a reference line.
- Keep `target_accept_prob` at $0.95$ or below. If NUTS needs more, improve the priors or the parametrization.
- For SVI, plot the ELBO trace. For a custom optimizer, use a step size schedule (`optax.cosine_decay_schedule`) and plot the objective trace.
- Take the uncertainty of a metric from the posterior (predictive) draws, not from a bootstrap. Compare two distributions with the same number of draws.
- When a choice is better a priori (a parametrization), make it and give the reason. Do not compare the alternatives in the notebook.

## No fake citations

Do not use fake citations. If you need to cite a source, ensure it exists.

## Keep ; after the end of a plot

In jupyter notebooks and Jupytext scripts, keep the `;` after the end of a plot.
