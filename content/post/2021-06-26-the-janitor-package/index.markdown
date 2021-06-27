---
title: The Janitor Package
author: Clancey Barrington
date: '2021-06-26'
slug: the-janitor-package
categories: [R, Data Cleaning]
tags: [R, Data Cleaning]
subtitle: ''
summary: 'An awesome R package that makes data cleaning simple.'
authors: []
lastmod: '2021-06-26T01:47:04-07:00'
featured: no
image:
  caption: Photo by Oliver Hale on Unsplash
  focal_point: ''
  preview_only: no
projects: []
---


When it comes to working with data you almost never have a clean data set. A good chunk of your time is usually spent on data cleaning. While this package won't solve all of your data cleaning woes, it can help make the process easier. Also it can be used with pipes %>%! Let's go through some of my favorite functions that I use often.  
  
`clean_names`:  
Often times the variable names in data sets come to us a little ugly. As a Python user, I try to avoid using periods as spaces and as a lazy person I try to avoid capitalization. This function will use '_' as a space and will make variable names lowercase.   
  

```r
# toy data set
dataset <- data.frame(
  No.Periods = rep("blah", 3),
  LOWERCASE = rep("la", 3),
  SpacePlease = rep("ha", 3)
  )

dataset
```

```
##   No.Periods LOWERCASE SpacePlease
## 1       blah        la          ha
## 2       blah        la          ha
## 3       blah        la          ha
```
  
Now lets apply the function and voila! Clean variable names.

```r
library(janitor)

dataset %>% 
  clean_names()
```

```
##   no_periods lowercase space_please
## 1       blah        la           ha
## 2       blah        la           ha
## 3       blah        la           ha
```
  
`row_to_names`:  
Recently I downloaded some data from a website and the column names were all messed up.The actual column names were in the first row and the current column names were unnecessary. That's where `row_to_names` comes in handy. We are able to set the column names to a specific row.


```r
dataset2 <- data.frame(
  Wrong = c("col1", 1, 2),
  X = c("col2", 3, 4)
)

dataset2
```

```
##   Wrong    X
## 1  col1 col2
## 2     1    3
## 3     2    4
```
  
After applying the function we see that the new column names are from row 1.

```r
dataset2 %>% 
  row_to_names(row_number = 1)
```

```
##   col1 col2
## 2    1    3
## 3    2    4
```
  
`remove_empty`:  
While you shouldn't always drop columns and rows with missing values. If the entire column or row is empty we should get rid of those. That's where `remove_empty` comes in handy.  

```r
dataset3 <- data.frame(
  col1 = rep(NA,3),
  col2 = c(1, NA, 4),
  col3 = c(4, NA, 5)
)

dataset3
```

```
##   col1 col2 col3
## 1   NA    1    4
## 2   NA   NA   NA
## 3   NA    4    5
```
  
After applying the function `col1` and row 2 are completely gone.

```r
dataset3 %>% 
  remove_empty()
```

```
##   col2 col3
## 1    1    4
## 3    4    5
```
  
Now let's use all three functions together.

```r
dataset4 <- data.frame(
  Wrong = c("col.1", 1, NA),
  X = c("Col 2", 3, NA),
  Y = c("col3", NA, NA)
)

dataset4
```

```
##   Wrong     X    Y
## 1 col.1 Col 2 col3
## 2     1     3 <NA>
## 3  <NA>  <NA> <NA>
```
  
As we can see `col3` is gone and only row 2 remains.

```r
dataset4 %>% 
  row_to_names(row_number = 1) %>% 
  clean_names() %>% 
  remove_empty()
```

```
##   col_1 col_2
## 2     1     3
```
  
These are some of the functions that I like to use from the janitor package. There are many more awesome functions to try out at  [CRAN](https://cran.r-project.org/web/packages/janitor/vignettes/janitor.html). Have fun cleaning your data!
