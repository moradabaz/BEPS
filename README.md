# BEPS
Simple Statistical Learning exercise studying a British Election Survey in 2002


---
title: "British vote prediction 2002"
author: 
  - Mourad Abbou Aazaz
---


```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
set.seed(1234)
```

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

# Variable analysis


Varible analysis is an important part of data science because we measure the influence and utilty of each of every variable on prediction result. 

We first take a look at dataset summary>

```{r}
summary(BEPS)
```

As we can see, there are no null values, so null values tratement won't be necessary. Now, we would like to have a look at the votes proportion>

```{r}
barplot(table(BEPS$vote))
```

As we see, the majority of the people survey claimed they would vote for the Labor Party. Now it's time to see what type of person would one of these three candidates. 


## Factor typed variables

Our two factor variables are `gender` and `vote`, which represent the voter gender and favorite party. Now, if we look at the table below, we can see that most males and females would vote for the Labor Party. 


```{r}
table(BEPS$gender, BEPS$vote)
```

To have a better visualization, we plot a spineplot:

```{r}
spineplot(BEPS[,10] ~ BEPS[, 1], data=BEPS, main="Gender/vote ratio", xlab = "Candidate party", ylab = "Gender", col = c("pink", "skyblue"))
```

Focusing on people that would vote for conservatives, there are slightly more females and males voting for this party, but the difference between males and females is pretty weak. To check if there's any statistical difference, we will execute a `chiSquare`.

```{r}
chisq.test(table(BEPS$gender, BEPS$vote))
```


The `p-value` is greater than `0.05`, which means differences are statistically insignificant and both factors are independent. 


## Numerical variables


Now we are going to explore the numerical predictor and analyze each of everyone of them.

#### Age

In the image below we can see that the majority of young people and people between 35 and 40 would vote for the Labour Party, while the elders would vote for the Conservative Party. The Liberal Democrat Party is not very popular though some people who are between 20 and 25, and people between 45 and 50 would for them. 


```{r}
spineplot(BEPS[,1] ~ BEPS[,2], data=BEPS, main = "Age/Vote Ratio", xlab = "Ages of People Surveyed", ylab = "Vote", col = c("red", "blue","green"))
```


\hfill \break

#### Knowledge on National Economy (economic.cond.national)

This variable shows how aware people are of national economic situation, where value 1 means a person knows nothing and 5 means a person is very aware of national economy. Well, the histogram below shows that majority of people aware of the nacional economic conditions and situation would vote for the Labour Party while the least they know about, they're more likely to vote for conservatives. 

```{r}
ggplot(BEPS) + aes(x=as.numeric(economic.cond.national), group=vote, fill=vote) + 
geom_bar(position = "stack") +
  geom_histogram(binwidth=0.25) +
coord_trans() +
scale_fill_manual(values = c("skyblue", "brown1", "orange")) + 
theme_classic()
```


#### Knowledge on Domestic Economy (economic.cond.national)

Now we are going to evaluate people's knowledge on domestic economy.


```{r}
ggplot(BEPS) + aes(x=as.numeric(economic.cond.household), group=vote, fill=vote) + 
geom_bar(position = "stack") +
  geom_histogram(binwidth=0.25) +
coord_trans() +
scale_fill_manual(values = c("skyblue", "brown1", "orange")) + 
theme_classic()
```

We obtain similar results to the ones we drew on National Economy. Basicly, people who are aware of national economy situation, are also aware of domestic economy. 





#### Conocimiento sobre política (political.knowledge) 

Now we'll look on people's political knowledge. 




```{r}
ggplot(BEPS) + aes(x=as.numeric(political.knowledge), group=vote, fill=vote) + 
geom_bar(position = "stack") +
  geom_histogram(binwidth=0.25) +
coord_trans() +
scale_fill_manual(values = c("skyblue", "brown1", "orange")) + 
theme_classic()
```

The results show us a funny thing. Most people who are aware of the economic situation would vote for Labour Party, but also people who have a low level of political knowledge do intend to vote for labor party. It would be interesting to study any correlation between people's economic awareness and their political knowledge. Meanwhile, as political knowledge grows, the vote intention is apparently more balanced.


#### Europe

Now let's see the level of euroscepticism voters have. 

```{r}
spineplot(BEPS[,1] ~ BEPS[,8], data=BEPS, main = "Political affinity and euroscepticism", 
          xlab = "Euroscepticism scale", ylab = "Vote", col = c("skyblue", "brown1", "orange"))
```

The results are clear. The more eurosceptic people are, the more like they are to vote for the Conservative Party while pro-europeans would vote for Labour Party. 



#### Blair

Let's focus on Blair Candidate. Blair was the leader of the Labour Party and candidate to Prime Minister in 2002. The `Blair` value shows the opinion people on Tony Blair, where value 1 represents the worst opinion and value 5 represents the best.


```{r}
ggplot(BEPS) + aes(x=as.numeric(Blair), group=vote, fill=vote) + 
geom_bar(position = "stack") +
  geom_histogram(binwidth=0.25) +
coord_trans() +
scale_fill_manual(values = c("skyblue", "brown1", "orange")) + 
theme_classic()
```

As expected, the better opinion people have on Blair, the more likely they are to vote for democrats. 


## Multivariable Analysis
