# BEPS
Simple Statistical Learning exercise studying a British Election Survey in 2002


---
title: "British vote prediction 2002"

---

# Introduction

In 2002 there was a survey which collected some information from people in the UK for the british elections in 2002. The information collected was stored in a data set called *BEPS* and registered data related to people's ideology, euroscepticism, opinion about certain candidates, etc. Having this information, our main target was to predict the candidate a person would vote if they gathered some of the opinions and ideology patterns. Thus, we made a statistical supervised learning analysis based on the data set we have in hand to study the influence and utility of each variable composing the dataset to predict the political party a person would vote.

# Data Set and Subsets

## Variables of the data set

The data set can be found using the `carData` library. 

```{r}
library(carData)
str(BEPS)
```

As we see, we have 10 different variables/predictors:

 * `vote`: This is the *output* we want to draw. It's a Factor variables which represent the three main political parties: Conservative, Liberal Democrat and Labour.
 * `age`: The age of each person surveyed.
 * `gender`: Each person's gender (Male or Female). 
 * `economic.cond.national`: This variable represents each person's knowledge of the national economy.
 * `economic.cond.household`: This variable represents each person's knowledge of families' household economy.
 * `Blair`: This variable represents each person's opinion about labourist candidate Blair.
 * `Hague`: This variable represents each person's opinion about conservative candidate Hague.
 * `Kennedy`: This variable represents each person's opinion about conservative candidate Kennedy.
 * `Europe`: This variable represents each person's euroscepticism. If a persons is very eurosceptic, the value will be 11. If is very pro-european, the value will be 0
 * `political.knowledge`: This variable represents each person's political knowledge.
 
## Data subsets

 The methodology to design a vote prediction model, following learning techniques, will be based on dividing the data set into two data subsets for training and testing, predictors analysis and several learning training algorithms trying to create the most reliable and accurate model. Our main target is to predict the **vote**, which is a factor variable and therefore our model must be a classification model.
 
Before starting to analyze the variables, we loaded the necessary libraries and create the subset partitions. 

```{r} 
library (reshape2)
library(lattice)
library(ggplot2)
library(caret)
library(mlbench)
library(e1071)
data(BEPS)      
BEPS.data.all <- BEPS
BEPS.data.outputs <- c("vote")
BEPS.data.inputs <- setdiff(names(BEPS.data.all), BEPS.data.outputs)
str(BEPS.data.inputs)
```

Now we have the following datasets:

* `BEPS.data.all` which represents the whole dataset.
* `BEPS.data.inputs` which contains all the values except the ones in the **vote** field.
* `BEPS.data.outputs` which contains the **vote** variable values.

We create a partition in which the 80% of the whole dataset will be used for training and the remaining 20% will be used for testing.

```{r}
train <-createDataPartition(BEPS.data.all[[BEPS.data.outputs]],p=0.8, list = FALSE, times = 1)
BEPS.data.all.80 <- BEPS.data.all[train,]
mask = sapply(BEPS.data.all.80, class) != "factor"
BEPS.data.all.20 <- BEPS.data.all[-train,]
BEPS.data.all.Train <- BEPS.data.all.80[,mask]
BEPS.data.all.Test <- BEPS.data.all[-train,]
BEPS.data.all.Test <- BEPS.data.all.Test[,mask]
output.values <- BEPS.data.all.80[[BEPS.data.outputs]]
```

The new subsets we created above are the following:

* `train` : The created partition. The 'p' value means the proportion and the argument `list = FALSE` se used so that the result can't be a list.
* `BEPS.data.all.80` : This subset will be used for training and it represents the 80% of the whole dataset.
* `BEPS.data.all.20` : This subset will be used for testing and it represents the 20% of the whole dataset.
* `BEPS.data.all.Train` : Training subset.
* `BEPS.data.all.Test` : Testing subset.
* `output.values` : Output values subset.
