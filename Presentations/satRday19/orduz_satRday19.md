<style>
.reveal .slides section .slideContent{
    font-size: 20pt;
}
</style>


Some Remedies for Several Class Imbalance
========================================================
author: Dr. Juan Orduz 
date: satRday Berlin  - 15.06.2019
autosize: true

Motivation  
========================================================

- Data Set Description
- Some Model Performance Metrics
- ...

Content
========================================================

- Data Set Description
- Some Model Performance Metrics
- ...


Data Set
========================================================



```r
data(AdultUCI, package = "arules")
raw_data <- AdultUCI

glimpse(raw_data, width = 65)
```

```
Observations: 48,842
Variables: 15
$ age              <int> 39, 50, 38, 53, 28, 37, 49, 52, 31, 4…
$ workclass        <fct> State-gov, Self-emp-not-inc, Private,…
$ fnlwgt           <int> 77516, 83311, 215646, 234721, 338409,…
$ education        <ord> Bachelors, Bachelors, HS-grad, 11th, …
$ `education-num`  <int> 13, 13, 9, 7, 13, 14, 5, 9, 14, 13, 1…
$ `marital-status` <fct> Never-married, Married-civ-spouse, Di…
$ occupation       <fct> Adm-clerical, Exec-managerial, Handle…
$ relationship     <fct> Not-in-family, Husband, Not-in-family…
$ race             <fct> White, White, White, Black, Black, Wh…
$ sex              <fct> Male, Male, Male, Male, Female, Femal…
$ `capital-gain`   <int> 2174, 0, 0, 0, 0, 0, 0, 0, 14084, 517…
$ `capital-loss`   <int> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0…
$ `hours-per-week` <int> 40, 13, 40, 40, 40, 40, 16, 45, 50, 4…
$ `native-country` <fct> United-States, United-States, United-…
$ income           <ord> small, small, small, small, small, sm…
```


```r
data_df <- model_list$functions$format_raw_data(df = raw_data)
```


Income Variable
========================================================
<img src="orduz_satRday19-figure/unnamed-chunk-4-1.png" title="plot of chunk unnamed-chunk-4" alt="plot of chunk unnamed-chunk-4" style="display: block; margin: auto;" />

Exploratory Data Analysis - Visualization
========================================================
<img src="orduz_satRday19-figure/unnamed-chunk-5-1.png" title="plot of chunk unnamed-chunk-5" alt="plot of chunk unnamed-chunk-5" style="display: block; margin: auto;" />

Feature Engineering
========================================================

$$
x \mapsto \log(x + 1)
$$

<img src="orduz_satRday19-figure/unnamed-chunk-6-1.png" title="plot of chunk unnamed-chunk-6" alt="plot of chunk unnamed-chunk-6" style="display: block; margin: auto;" />


Data Preparation
========================================================

```r
df <- data_df %>% 
  mutate(capital_gain_log = log(`capital-gain` + 1), 
         capital_loss_log = log(`capital-loss` + 1)) %>% 
  select(- `capital-gain`, - `capital-loss`) %>% 
  drop_na()

# Define observation matrix and target vector. 
X <- df %>% select(- income)
y <- df %>% pull(income)

# Add dummy variables. 
dummy_obj <- dummyVars("~ .", data = X, sep = "_")

X <- predict(object = dummy_obj, newdata = X) %>% as_tibble()

# Remove predictors with near zero variance. 
cols_to_rm <- colnames(X)[nearZeroVar(x = X, freqCut = 5000)]
  
X %<>% select(- cols_to_rm) 
```

Data Split
========================================================

```r
# Split train - other
split_index_1 <- createDataPartition(y = y, p = 0.7)$Resample1

X_train <- X[split_index_1, ]
y_train <- y[split_index_1]

X_other <- X[- split_index_1, ]
y_other <- y[- split_index_1]

split_index_2 <- createDataPartition(y = y_other, p = 1/3)$Resample1

# Split evaluation - test
X_eval <- X_other[split_index_2, ]
y_eval <- y_other[split_index_2]

X_test <- X_other[- split_index_2, ]
y_test <- y_other[- split_index_2]
```


Confision Matrix
========================================================

<table class="table" style="margin-left: auto; margin-right: auto;">
 <thead>
  <tr>
   <th style="text-align:left;">   </th>
   <th style="text-align:center;"> Condition Positive </th>
   <th style="text-align:center;"> Condition Negative </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> Prediction Positive </td>
   <td style="text-align:center;"> TP </td>
   <td style="text-align:center;"> FP </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Prediction Negative </td>
   <td style="text-align:center;"> FN </td>
   <td style="text-align:center;"> TN </td>
  </tr>
</tbody>
</table>

  - TP = True Positive
  - TN = True Negative 
  - FP = False Positive
  - FN = False Negative
  - N = TP + TN + FP + FN
  
Performance Metrics
========================================================

- Accuracy 

$$
\text{acc} = \frac{TP + TN}{N}
$$

- [Kappa](https://en.wikipedia.org/wiki/Cohen%27s_kappa) 

$$
\kappa = \frac{p_o - p_e}{1 - p_e}
$$

where 

  - $p_e$ = Expected Accuracy (random chance).
  - $p_o$ = Observed Accuracy. 
  
The kappa metric can be thought as a modification of the accuracy metric based on the class proportions. 

Performance Metrics
========================================================

- [Sensitivity](https://en.wikipedia.org/wiki/Sensitivity_and_specificity) (= [Recall](https://en.wikipedia.org/wiki/Precision_and_recall))

$$
\text{sens} = \frac{TP}{TP + FN}
$$

- [Specitivity](https://en.wikipedia.org/wiki/Sensitivity_and_specificity)

$$
\text{spec} = \frac{TN}{TN + FP}
$$

- [Precision](https://en.wikipedia.org/wiki/Precision_and_recall)

$$
\text{prec} = \frac{TP}{TP + FP}
$$

- AUC

This metric refers to the area under the curve of the [ROC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) curve. It does not depend on the cutoff probability threshold for the predicted classes. 

Machine Learning Models
========================================================

0. Trivial Model
1. Partial Least Squares + Logistic Regression
2. Stochastic Gradient Boosting

Trivial Model
========================================================

We predict the same class `income` = `small`


```r
y_pred_trivial <- map_chr(.x = y_test, .f = ~ "small") %>% 
  as_factor(ordered = TRUE, levels = c("small", "large"))
```

We compute the confusion matrix to get the metrics.


```r
# Confusion Matrix. 
conf_matrix_trivial <-  confusionMatrix(data = y_pred_trivial, 
                                        reference =  y_test)
```

<table class="table" style="margin-left: auto; margin-right: auto;">
 <thead>
  <tr>
   <th style="text-align:center;"> term </th>
   <th style="text-align:center;"> estimate </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:center;"> accuracy </td>
   <td style="text-align:center;"> 0.751 </td>
  </tr>
  <tr>
   <td style="text-align:center;"> kappa </td>
   <td style="text-align:center;"> 0.000 </td>
  </tr>
  <tr>
   <td style="text-align:center;"> sensitivity </td>
   <td style="text-align:center;"> 1.000 </td>
  </tr>
  <tr>
   <td style="text-align:center;"> specificity </td>
   <td style="text-align:center;"> 0.000 </td>
  </tr>
</tbody>
</table>

Trivial Model - ROC
========================================================

<img src="orduz_satRday19-figure/unnamed-chunk-13-1.png" title="plot of chunk unnamed-chunk-13" alt="plot of chunk unnamed-chunk-13" style="display: block; margin: auto;" />
