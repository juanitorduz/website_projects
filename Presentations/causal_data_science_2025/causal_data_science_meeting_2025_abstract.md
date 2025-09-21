# Bridging the Causal Inference Gap: A Hands-On Workshop Using Simulations and Probabilistic Programming

[**Dr. Juan Orduz**](https://juanitorduz.github.io/)

## Workshop Abstract

One of the most significant barriers to implementing causal inference in industry is not technical complexity, but rather organizational resistance rooted in unfamiliarity with causal thinking. While some (not all) data scientists understand the mathematical foundations of causal inference, translating these concepts to non-technical stakeholders—executives, product managers, and domain experts—remains a persistent challenge. This hands-on workshop teaches participants how to use simulation-based approaches with probabilistic programming languages as powerful pedagogical tools to demonstrate causal concepts, build organizational understanding, and secure buy-in for causal inference initiatives.

Simulations prove particularly valuable in scenarios where traditional A/B testing falls short. Consider marketing funnel effects, where upper-funnel brand marketing (TV advertising) influences lower-funnel performance marketing (Google Ads), creating complex confounding relationships. Using probabilistic programming, we can simulate these multi-channel attribution problems and demonstrate how naive correlation analysis might incorrectly attribute sales increases to Google Ads when the true driver is TV advertising.

Similarly, in logistics and operations research, switchback experiments—where interventions are applied across time periods or geographic regions rather than individual users—present unique analytical challenges. Simulations can illustrate how temporal correlation and spillover effects complicate causal inference, helping stakeholders understand why specialized methods are needed beyond simple before-and-after comparisons.

### The Power of Explicit Data Generation

Traditional approaches to explaining causal inference often rely on abstract theoretical frameworks that can alienate non-technical audiences. By contrast, simulations with explicit data generating processes provide intuitive, visual demonstrations of causal relationships. Using probabilistic programming languages like NumPyro and PyMC, we can construct transparent models where the true causal structure is known by design, enabling stakeholders to witness firsthand how confounders bias naive analyses and how proper causal methods recover true effects.

### Leveraging Observe and Do Operators

Modern probabilistic programming frameworks offer unique advantages through their `observe` and `do` operators, which directly map to Pearl's causal hierarchy. The `observe` operator demonstrates passive observation (correlation), while the `do` operator showcases active intervention (causation). This clear distinction helps non-technical stakeholders understand why "correlation is not causation" and why A/B tests and observational studies require different analytical approaches.

### Workshop Structure

This interactive workshop will equip participants with practical skills and tools for using simulation-based education to drive causal inference adoption in their organizations:

**Part 1: Building Intuition Through Simulation (45 minutes)**
- Hands-on exercise: Participants will create simple simulations using NumPyro/PyMC to generate data with known causal structures
- Live coding demonstration: Building intuition about confounders and selection bias through explicit data generating processes
- Interactive exercise: Comparing naive correlation analysis with proper causal methods on simulated data

**Part 2: Collaborative DAG Construction (45 minutes)**
- Small group activity: Participants work in teams to draft causal DAGs relevant to their domains (marketing, logistics, product, etc.)
- Cross-team sharing: Groups present their DAGs and discuss different perspectives on causal relationships
- Facilitated discussion: How causal directions differ across domains and stakeholder viewpoints

**Part 3: Practical Case Study Implementation (60 minutes)**
- Guided implementation: Participants choose one component from their DAGs to model and simulate
- Focus areas: Confounders, selection bias, or temporal effects relevant to their use cases
- Real-world application: Using actual data patterns to simulate future scenarios and strategy changes

**Part 4: Strategic Implementation Planning (30 minutes)**
- Group exercise: Developing actionable plans for introducing causal inference in participants' organizations
- Best practices discussion: Starting with incremental wins rather than perfect solutions
- Resource sharing: Templates and code examples for ongoing simulation-based education efforts

### Prerequisites
- Basic Python programming experience
- Familiarity with data analysis concepts (no prior causal inference knowledge required)
- Laptop with Python environment (setup instructions will be provided in advance)
