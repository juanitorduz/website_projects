---
marp: true
theme: default
style: |
    img[alt~="center"] {
      display: block;
      margin: auto;
    }
---

# Introduction to Uplift Modeling

## [Dr. Juan Orduz](https://juanitorduz.github.io/)


[PyConDE & PyData Berlin 2022](https://2022.pycon.de/)

![w:200 center](images/logo.png)

---
<!--
_footer: Image taken from https://www.uplift-modeling.com/en/latest/user_guide/introduction/clients.html 
-->

# Motivation

## How can we optimally select customers to be treated by marketing incentives?

![w:400 center](https://www.uplift-modeling.com/en/latest/_images/ug_clients_types.jpg)

---

# We can not **send** and **not send** incentives to the same customers at the same time

![w:600 center](images/two-spiderman.jpeg)



---

# What is Uplift Modeling?

From [Gutierrez, P., & Gérardy, J. Y. (2017). "Causal Inference and Uplift Modelling: A Review of the Literature"](https://proceedings.mlr.press/v67/gutierrez17a/gutierrez17a.pdf)

 > - Uplift modeling refers to the set of techniques used to model the incremental impact of an action or treatment on a customer outcome.
 > - Uplift modeling is therefore both a Causal Inference problem and a Machine Learning one. 
 
---
<!--
_footer: Taken from [Gutierrez, P., & Gérardy, J. Y. (2017). "Causal Inference and Uplift Modelling: A Review of the Literature"](https://proceedings.mlr.press/v67/gutierrez17a/gutierrez17a.pdf)
-->

# Conditional Average Treatment Effect

- Let $Y^{1}_{i}$ denote person $i$'s outcome when it receives the treatment and $Y^{0}_{i}$ when it does not receive the treatment.
- We are interested in understanding the *causal effect* $Y^{1}_{i} - Y^{0}_{i}$ and the  *conditional average treatment effect* $CATE = E[Y^{1}_{i} | X_{i}] - E[Y^{0}_{i} | X_{i}]$, where $X_{i}$ is a feature vector of the $i$-th person.
- **However, we can not observe them!** 🙁 

---
<!--
_footer: Taken from [Gutierrez, P., & Gérardy, J. Y. (2017). "Causal Inference and Uplift Modelling: A Review of the Literature"](https://proceedings.mlr.press/v67/gutierrez17a/gutierrez17a.pdf)
-->
# Uplift

Let $W_{i}$ is a binary variable indicating whether person $i$ received the treatment, so that

$$Y_{i}^{obs} = Y^{1}_{i} W_{i} + (1 - W_{i}) Y^{0}_{i}$$

## Unconfoundedness Assumption

If we **assume** that the treatment assignment $W_{i}$ is independent of $Y^{1}_{i}$  and $Y^{0}_{i}$ conditional on $X_i$, then we can estimate the $CATE$ from observational data by computing the empirical counterpart:

$$\text{\bf{uplift}} = \widehat{CATE} = E[Y_{i} | X_{i}, W_{i}=1] - E[Y_{i} | X_{i}, W_{i}=0]$$

---

# S-Learner
<!--
_footer: Taken from https://www.uplift-modeling.com/en/latest/user_guide/models/index.html
-->

### Step 1: Training

$$
\underbrace{
\left(
\begin{array}{cccc}
x_{11} & \cdots & x_{1k} & w_{1} \\
\vdots & \ddots & \vdots & \vdots \\
x_{11} & \cdots & x_{nk} & w_{n} \\
\end{array}
\right)}_{X\bigoplus W}
\xrightarrow{f}
\left(
\begin{array}{c}
y_{1} \\
\vdots \\
y_{n}
\end{array}
\right)
$$

### Step 2: Uplift Prediction

$$
\hat{f}\left(
\begin{array}{cccc}
x_{11} & \cdots & x_{1k} & 1 \\
\vdots & \ddots & \vdots & \vdots \\
x_{11} & \cdots & x_{mk} & 1 \\
\end{array}
\right)
-
\hat{f}
\left(
\begin{array}{cccc}
x_{11} & \cdots & x_{1k} & 0 \\
\vdots & \ddots & \vdots & \vdots \\
x_{11} & \cdots & x_{mk} & 0 \\
\end{array}
\right)
$$

---
<!--
_footer: Taken from https://causalml.readthedocs.io/en/latest/methodology.html#meta-learner-algorithms
-->

# T-Learner

### Step 1: Training

$$
\underbrace{
\left(
\begin{array}{ccc}
x_{11} & \cdots & x_{1k} \\
\vdots & \ddots & \vdots \\
x_{11} & \cdots & x_{n_{C}k} \\
\end{array}
\right)}_{X|_{\text{control}}}
\xrightarrow{f_{C}}
\left(
\begin{array}{c}
y_{1} \\
\vdots \\
y_{n_{C}}
\end{array}
\right)
$$

$$
\underbrace{
\left(
\begin{array}{ccc}
x_{11} & \cdots & x_{1k}  \\
\vdots & \ddots & \vdots \\
x_{11} & \cdots & x_{n_{T}k} \\
\end{array}
\right)}_{X |_{\text{treatment}}}
\xrightarrow{f_{T}}
\left(
\begin{array}{c}
y_{1} \\
\vdots \\
y_{n_{T}}
\end{array}
\right) 
$$

---
<!--
_footer: Taken from https://causalml.readthedocs.io/en/latest/methodology.html#meta-learner-algorithms
-->

# T-Learner

### Step 2: Uplift Prediction

$$
\hat{f}_{T}\left(
\begin{array}{cccc}
x_{11} & \cdots & x_{1k} \\
\vdots & \ddots & \vdots \\
x_{11} & \cdots & x_{mk} \\
\end{array}
\right)
-
\hat{f}_{C}
\left(
\begin{array}{cccc}
x_{11} & \cdots & x_{1k} \\
\vdots & \ddots & \vdots \\
x_{11} & \cdots & x_{mk} \\
\end{array}
\right)
$$

---
<!--
_footer: Taken from https://causalml.readthedocs.io/en/latest/methodology.html#meta-learner-algorithms
-->

# X-Learner

### Step 1: Training: Same as T-Learner

### Step 2: Compute imputed treatment effects

$$
\hat{D}^{T} \coloneqq
\left(
\begin{array}{c}
y_{1} \\
\vdots \\
y_{n_{C}}
\end{array}
\right)
- 
\hat{f}_{C}
\left(
\begin{array}{cccc}
x_{11} & \cdots & x_{1k} \\
\vdots & \ddots & \vdots \\
x_{11} & \cdots & x_{n_{C}k} \\
\end{array}
\right)
$$

$$
\hat{D}^{C} \coloneqq
\hat{f}_{Y}
\left(
\begin{array}{cccc}
x_{11} & \cdots & x_{1k} \\
\vdots & \ddots & \vdots \\
x_{11} & \cdots & x_{n_{T}k} \\
\end{array}
\right)
-
\left(
\begin{array}{c}
y_{1} \\
\vdots \\
y_{n_{C}}
\end{array}
\right)
$$
---

# Some python implementations

- [`causalml`](https://github.com/uber/causalml)

![w:400 center](https://raw.githubusercontent.com/uber/causalml/master/docs/_static/img/logo/causalml_logo.png)

- [`scikit-uplift`](https://github.com/maks-sh/scikit-uplift)

![w:700 center](https://raw.githubusercontent.com/maks-sh/scikit-uplift/dev/docs/_static/sklift-github-logo.png)

---

## References:

- [Gutierrez, P., & Gérardy, J. Y. (2017). "Causal Inference and Uplift Modelling: A Review of the Literature"](https://proceedings.mlr.press/v67/gutierrez17a/gutierrez17a.pdf)

- [Karlsson, H. (2019) "Uplift Modeling: Identifying Optimal Treatment Group Allocation and Whom to Contact to Maximize Return on Investment"](http://www.diva-portal.org/smash/get/diva2:1328437/FULLTEXT01.pdf)

---

# Thank you!

## More Info: [juanitorduz.github.io/](https://juanitorduz.github.io/)

![w:400 center](images/qr-code-juanitorduz.png)
