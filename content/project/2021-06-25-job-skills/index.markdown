---
title: Job Skills Project
author: Clancey Barrington
date: '2021-06-25'
slug: job-skills
categories: []
tags: [R, NLP]
subtitle: ''
summary: 'Analyzing the most desired job skills for popular data science positions.'
authors: []
lastmod: '2021-06-25T13:40:04-07:00'
featured: no
image:
  caption: 'Photo by Firmbee.com on Unsplash'
  focal_point: ''
  preview_only: no
projects: []
---


This is a project that I did back in my Fall 2020 semester for a research class. As a soon to be graduate at the time, I was curious what skills I still needed to learn. The data is from [Kaggle](https://www.kaggle.com/andrewmvd/business-analyst-jobs) and includes Glassdoor job postings. We will be doing some text mining with the help of [Text Mining with R: A Tidy Approach](https://www.tidytextmining.com/tidytext.html).  
  
### Data Cleaning and Wrangling
Before we get to the good stuff we will read in and clean our data.

```r
data_analyst <- read.csv("DataAnalyst.csv")
data_scientist <- read.csv("DataScientist.csv")
data_engineer <- read.csv("DataEngineer.csv")
business_analyst <- read.csv("BusinessAnalyst.csv")
```

The first thing we need to take of is the business analyst data. For some reason some of the Florida locations are shifted two columns to the left.

```r
library(tidyverse)

business_analyst %>% 
  select(Rating, Location) %>% 
  filter(Rating == "Jacksonville, FL") %>% 
  head(3)
```

```
##             Rating         Location
## 1 Jacksonville, FL 10000+ employees
## 2 Jacksonville, FL 10000+ employees
## 3 Jacksonville, FL 10000+ employees
```
  
We'll make a vector of the locations that are messed up, fix that section, and add it back to the .

```r
remove <- c("Jacksonville, FL", "Jacksonville Beach, FL", "Orange Park, FL", 
            "Ponte Vedra Beach, FL", "Mayport, FL", "Fleming Island, FL")

# rename columns so everything matches
ba_FL<-  business_analyst %>% 
            filter(Rating %in% remove) %>% 
            select(-c(Competitors, Easy.Apply)) %>% 
            rename(Job.Title = X, Salary.Estimate = index, Job.Description = Job.Title,
                  Rating = Salary.Estimate, Company.Name = Job.Description,
                  Location = Rating, Headquarters = Company.Name, Size = Location,
                  Founded = Headquarters, Type.of.ownership = Size, Industry = Founded,
                  Sector = Type.of.ownership, Revenue = Industry, Competitors = Sector,
                  Easy.Apply = Revenue)

# add back to business
ba_noFL <- business_analyst %>%
  filter(!Rating %in% remove) %>% 
  select(-c(X, index)) 

business_analyst <- rbind(ba_noFL, ba_FL)
```
  
Let's double check that everything is good.

```r
business_analyst %>% 
  select(Rating, Location) %>% 
  filter(Location == "Jacksonville, FL") %>% 
  head(3)
```

```
##   Rating         Location
## 1    3.3 Jacksonville, FL
## 2    2.3 Jacksonville, FL
## 3    2.3 Jacksonville, FL
```
  
Now that the business analyst data is good we will make a  with all of the job postings.

```r
# remove unneeded columns
data_analyst <- data_analyst %>% 
  select(-c(X, Competitors))
data_scientist <- data_scientist %>% 
  select(-c(X, index, Competitors))
data_engineer <- data_engineer %>% 
  select(-c(Competitors))
business_analyst <- business_analyst %>% 
  select(-c(Competitors))

# add job posting category to 
data_analyst['job_type'] = 'data analyst'
data_scientist['job_type'] = 'data scientist'
data_engineer['job_type'] = 'data engineer'
business_analyst['job_type'] = 'business analyst'

# combine data sets
jobs_data <- rbind(data_analyst, data_scientist, data_engineer, business_analyst)
```
  
As a python user, I'm not a fan of the variable names using periods as spaces, so we will use the janitor package to tidy everything up. 

```r
# fix column names don't like periods
library(janitor)

jobs_data <- jobs_data %>%
  clean_names()
```
  
Now we get to deal with missing values and replace them with NA.

```r
library(naniar)

jobs_data$rating <- as.double(jobs_data$rating)

# find missing data and replace with NA
jobs_data <- jobs_data %>%
  replace_with_na(replace = list(salary_estimate = "-1", 
                                 rating = "-1",
                                 company_name = c("", "1"),
                                 location = "Unknown",
                                 headquarters = "-1",
                                 size = c("-1", "Unknown"),
                                 founded = "-1",
                                 type_of_ownership = c("-1", "Unknown"),
                                 industry = c("-1", "Unknown / Non-Applicable"),
                                 sector = "-1",
                                 revenue = c("-1", "Unknown / Non-Applicable", "True")
                                 ))
```

Now let's do a little more cleaning.

```r
# fix up the values in the data
# salary estimate remove (Glassdoor est.) and Empy
# remove rating from company name
jobs_data <- jobs_data %>% 
  mutate(salary_estimate = as.character(gsub("[(Glassdoor est.)]", "", salary_estimate))) %>% 
  mutate(salary_estimate = as.character(gsub("Empy", "", salary_estimate))) %>% 
  mutate(company_name = as.character(gsub("[[:digit:]]+$", "", company_name))) %>% 
  mutate(company_name = as.character(gsub("[[:punct:]]+$", "", company_name))) %>% 
  mutate(company_name = as.character(gsub("[[:digit:]]+$", "", company_name))) %>% 
  mutate(company_name = as.character(gsub("\n", "", company_name)))
```

We are originally given an interval for estimated salary. To help with our analysis we will use the midpoint of the salary range.

```r
jobs_data <- jobs_data %>%
  mutate(desc_number = row_number()) %>% 
  #get rid of hourly pay
  filter(!grepl("PHu",salary_estimate)) %>% 
  # make salary numeric
  # make salary range only have numbers
  mutate(salary = as.numeric(gsub("[^\\d]+", "", salary_estimate, perl=TRUE))) %>% 
  # get the min, max, and mid salary
  mutate(min = ifelse(nchar(salary) == 4, floor(salary / 100), 
                ifelse(nchar(salary) == 5, floor(salary / 1000), floor(salary / 1000)))) %>% 
  mutate(max = ifelse(nchar(salary) == 4, salary %% 100, 
                ifelse(nchar(salary) == 5, salary %% 1000, salary %% 1000))) %>% 
  mutate(mid = (min + max) / 2) 
```
  
Lastly, we get the state for the job posting.

```r
# get the state
jobs_data$state <- str_sub(jobs_data$location,-2,-1)
jobs_data$state <- as.factor(jobs_data$state)

# remove united kingdom rows
jobs_data<- jobs_data %>%
  filter(!grepl("om",state))
```

### Text Mining
Alright, let's do some text mining! First, we will spruce up the job descriptions, then we will tokenize and remove stop words. Also note that we included a job posting number to prevent double counting words from the same job description.

```r
library(tidytext)

# was coded as a factor needs to be a character
jobs_data$job_description <- as.character(jobs_data$job_description)

job_desc_words <- jobs_data %>% 
  select(job_description, job_type)

# â and Â instead of spaces and put a space for / because R/Python was listed in some
job_desc_words <- job_desc_words %>% 
  mutate(job_description = as.character(gsub("[â Â / .]+", " ", job_description)))

# add description number so I have no duplicate words in a listing 
original_desc <- job_desc_words %>%
  mutate(desc_number = row_number()) %>%
  ungroup() 

tidy_desc <- original_desc %>%
  unnest_tokens(word, job_description) 

data(stop_words)

# stopwords had r and c in it so I removed it
stop_words <- stop_words[!stop_words$word == "r",]
stop_words <- stop_words[!stop_words$word == "c",]

# take out stop words
tidy_desc <- tidy_desc %>%
  anti_join(stop_words) 

# remove duplicate rows
tidy_desc <- tidy_desc[!duplicated(tidy_desc), ]
```

Now we get to look at the top 10 words and well it's not too exciting.

```r
tidy_desc %>%
  count(word, sort = TRUE) %>%
  filter(n > 7450) %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(word, n)) +
  geom_col(fill = "#3a6a8a") +
  xlab(NULL) +
  coord_flip()+
  labs(title = "Top 10 Words in Job Descriptions", y = "Count")
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-12-1.png" width="672" />
How about we look at a word cloud with the top 100 words. A little bit more interesting.

```r
library(wordcloud)
library(RColorBrewer)

tidy_desc2 <- tidy_desc

pal <- brewer.pal(9,"Paired")

tidy_desc2$word <- gsub("[0-9]+", "", tidy_desc2$word)

tidy_desc2 %>%
  count(word) %>%
  with(wordcloud(word, n, max.words = 100, colors = pal))
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-13-1.png" width="672" />

Let's have a little bit more fun and look at some bigrams.

```r
job_bigrams <- original_desc %>%
  unnest_tokens(bigram, job_description, token = "ngrams", n = 2)

#separate bigrams to remove stop words
bigrams_separated <- job_bigrams %>%
  separate(bigram, c("word1", "word2"), sep = " ")

bigrams_filtered <- bigrams_separated %>%
  filter(!word1 %in% stop_words$word) %>%
  filter(!word2 %in% stop_words$word)

# new bigram counts:
bigram_counts <- bigrams_filtered %>% 
  count(word1, word2, sort = TRUE)

# unite the words
bigrams_united <- bigrams_filtered %>%
  unite(bigram, word1, word2, sep = " ")
```

Let's look at the bigram graph. In it we can see machine learning, communication skills, and fast paced.

```r
library(igraph)
library(ggraph)

#create graph
bigram_graph <- bigram_counts %>%
  filter(n > 1500) %>%
  graph_from_data_frame()

# graph it

set.seed(2016)

a <- grid::arrow(type = "closed", length = unit(.15, "inches"))

ggraph(bigram_graph, layout = "fr") +
  geom_edge_link(aes(edge_alpha = n), show.legend = FALSE,
                 arrow = a, end_cap = circle(.07, 'inches')) +
  geom_node_point(color = "lightblue", size = 5) +
  geom_node_text(aes(label = name), vjust = 1, hjust = 1) +
  theme_void()
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-15-1.png" width="672" />
  
When we look at the word cloud we can see lots to do with data like data warehouse and data driven.

```r
set.seed(78)
bigrams_united %>%
  count(bigram) %>%
  with(wordcloud(bigram, n, max.words = 50, colors = pal))
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-16-1.png" width="672" />
Two words are fun, but what about three?

```r
job_trigrams <- original_desc %>%
  unnest_tokens(trigram, job_description, token = "ngrams", n = 3)

# separate to remove stop words
trigrams_separated <- job_trigrams %>%
  separate(trigram, c("word1", "word2", "word3"), sep = " ")

trigrams_filtered <- trigrams_separated %>%
  filter(!word1 %in% stop_words$word) %>%
  filter(!word2 %in% stop_words$word) %>% 
  filter(!word3 %in% stop_words$word)

# new trigram counts:
trigram_counts <- trigrams_filtered %>% 
  count(word1, word2, word3, sort = TRUE)

# put all back together
trigrams_united <- trigrams_filtered %>%
  unite(trigram, word1, word2, word3, sep = " ") 
```
  
Our trigram graph shows us even more. We can see natural language and data visualization.

```r
# make the graph
trigram_graph <- trigram_counts %>%
  filter(n > 300) %>%
  graph_from_data_frame()

# graph it
set.seed(2016)

a <- grid::arrow(type = "closed", length = unit(.15, "inches"))

ggraph(trigram_graph, layout = "fr") +
  geom_edge_link(aes(edge_alpha = n), show.legend = FALSE,
                 arrow = a, end_cap = circle(.07, 'inches')) +
  geom_node_point(color = "lightblue", size = 5) +
  geom_node_text(aes(label = name), vjust = 1, hjust = 1) +
  theme_void()
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-18-1.png" width="672" />
  
Most of the trigram word cloud has to do with equal employment, but if we look around again machine learning, data visualization, and communication pop up.

```r
set.seed(78)
trigrams_united %>%
  count(trigram) %>%
  with(wordcloud(trigram, n, max.words = 50, colors = pal))
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-19-1.png" width="672" />
  
#### Soft Skills
It's been fun looking the most common words and pairings, but now lets look even deeper at the desired soft/ hard skills, languages, and tools. First up, soft skills.

```r
# look for occurrence of word in tidy_desc if one word
# look for occurrence of word in the bigram if two words
comm <- tidy_desc %>% 
  filter(str_detect(word, 'communication')) %>% 
  rename(bigram = word)
comm['skill'] = 'communication'

proj_man <- bigrams_united %>%
  filter(str_detect(bigram, "project management"))
proj_man['skill'] = 'project management'

analytical <- tidy_desc %>%
  filter(str_detect(word, "analytical"))%>% 
  rename(bigram = word)
analytical['skill'] = 'analytical'

interpersonal <- tidy_desc %>%
  filter(str_detect(word, "interpersonal"))%>% 
  rename(bigram = word)
interpersonal['skill'] = 'interpersonal'

time_man <- bigrams_united %>%
  filter(str_detect(bigram, "time manage"))
time_man['skill'] = 'time management'

# look for different ways a phrase can be said
crit_think1 <- bigrams_united %>%
  filter(str_detect(bigram, "critical think"))
crit_think2 <- bigrams_united %>%
  filter(str_detect(bigram, "think critically"))
crit_think3 <- bigrams_united %>%
  filter(str_detect(bigram, "critically think"))
crit_think <- rbind(crit_think1, crit_think2, crit_think3)     
crit_think['skill'] = 'critical thinking'

organization1 <- bigrams_united %>%
  filter(str_detect(bigram, "organization skill"))
organization2 <- tidy_desc %>%
  filter(str_detect(word, "organized")) %>% 
  rename(bigram = word)
organization3 <- bigrams_united %>%
  filter(str_detect(bigram, "organizational skill"))
organization <- rbind(organization1, organization2, organization3)
organization['skill'] = 'organization'

creative <- tidy_desc %>%
  filter(str_detect(word, "creativ"))%>% 
  rename(bigram = word)
creative['skill'] = 'creativity'

leadership <- tidy_desc%>%
  filter(str_detect(word, "leadership"))%>% 
  rename(bigram = word)
leadership['skill'] = 'leadership'

# put them all together
soft_skills <- rbind(comm, proj_man, analytical, interpersonal, time_man, crit_think, organization, creative, leadership)

# get the desc number and skill only to remove duplicates
soft_skills <- soft_skills %>% 
  select(desc_number, skill)
soft_skills <- soft_skills[!duplicated(soft_skills), ]

# get percentage
percent_soft <- soft_skills %>% 
  group_by(skill) %>% 
  summarise(n = n())  %>% 
  mutate(proportion = n / nrow(jobs_data)) 
```
  
For soft skills communication and analytical come out on top.

```r
# plot bar graph
percent_soft %>% 
  ggplot(aes(x = skill, y = proportion)) +
  geom_bar(stat = "identity", fill = "#492a8c") +
  labs(title = "Most Desired Soft Skills") +
  coord_flip()
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-21-1.png" width="672" />
  
#### Hard Skills
Next up, hard skills.

```r
comp_sci1 <- bigrams_united %>%
  filter(str_detect(bigram, "computer science"))
comp_sci2 <- bigrams_united %>%
  filter(str_detect(bigram, "comp sci"))
comp_sci <- rbind(comp_sci1, comp_sci2)
comp_sci['skill'] = 'computer science'

mach_learn1 <- bigrams_united %>%
  filter(str_detect(bigram, "machine learn"))
mach_learn2 <- tidy_desc %>%
  filter(word == "ml") %>% 
  rename(bigram = word)
mach_learn <- rbind(mach_learn1, mach_learn2)
mach_learn['skill'] = 'machine learning'

business_int <- bigrams_united %>%
  filter(str_detect(bigram, "business intel"))
business_int['skill'] = 'business intelligence'

data_analysis <- bigrams_united %>%
  filter(str_detect(bigram, "data analysis"))
data_analysis['skill'] = 'data analysis'

statistics <- tidy_desc %>%
  filter(str_detect(word, "statistics")) %>% 
  rename(bigram = word)
statistics['skill'] = 'statistics'

data_vis <- bigrams_united %>%
  filter(str_detect(bigram, "data visual"))
data_vis['skill'] = 'data visualization'

programming <- tidy_desc %>%
  filter(str_detect(word, "programming")) %>% 
  rename(bigram = word)
programming['skill'] = 'programming'

soft_eng <- bigrams_united %>%
  filter(str_detect(bigram, "software engineer"))
soft_eng['skill'] = 'software engineering'

hard_skills <- rbind(comp_sci, mach_learn, business_int, data_analysis, statistics, data_vis, programming, soft_eng)
hard_skills <- hard_skills %>% 
  select(desc_number, skill)
hard_skills <- hard_skills[!duplicated(hard_skills), ]

percent_hard <- hard_skills %>% 
  group_by(skill) %>% 
  summarise(n = n())  %>% 
  mutate(proportion = n / nrow(jobs_data))
```
  
For hard skills it looks like computer science and programming come out on top.

```r
percent_hard %>% 
  ggplot(aes(x = skill, y = proportion)) +
  geom_bar(stat = "identity", fill = "#492a8c") +
  labs(title = "Most Desired Hard Skills") +
  coord_flip()
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-23-1.png" width="672" />
  
#### Languages
On to languages!

```r
r <- tidy_desc %>%
  filter(word == "r") 
r['skill'] = 'r'

python <- tidy_desc %>%
  filter(str_detect(word, "python"))
python['skill'] = 'python'

c <- tidy_desc %>%
  filter(word == "c") 
c['skill'] = 'c++/ c'

java <- tidy_desc %>%
  filter(word == "java")
java['skill'] = 'java'

sas <- tidy_desc %>%
  filter(word == "sas")
sas['skill'] = 'sas'

scala <- tidy_desc %>%
  filter( word == "scala")
scala['skill'] = 'scala'

sql <- tidy_desc %>%
  filter(str_detect(word, "sql"))
sql['skill'] = 'sql'

lang_skills <- rbind(r, python, c, java, sas, scala, sql)
lang_skills <- lang_skills %>% 
  select(desc_number, skill)
lang_skills <- lang_skills[!duplicated(lang_skills), ]

percent_lang <- lang_skills %>% 
  group_by(skill) %>% 
  summarise(n = n())  %>% 
  mutate(proportion = n / nrow(jobs_data)) 
```
  
For languages SQL dominates with python behind it.

```r
percent_lang %>% 
  ggplot(aes(x = skill, y = proportion)) +
  geom_bar(stat = "identity", fill = "#35785f") +
  labs(title = "Most Desired Language") +
  coord_flip()
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-25-1.png" width="672" />


#### Tools
Last up is tools.

```r
tableau <- tidy_desc %>%
  filter(str_detect(word, "tableau"))
tableau['skill'] = 'tableau'

excel <- tidy_desc %>%
  filter(word == "excel")
excel['skill'] = 'excel'

microsoft <- tidy_desc %>%
  filter(str_detect(word, "microsoft")) 
microsoft['skill'] = 'microsoft'

hadoop <- tidy_desc %>%
  filter(str_detect(word, "hadoop"))
hadoop['skill'] = 'hadoop'

spark <- tidy_desc %>%
  filter(str_detect(word, "spark"))
spark['skill'] = 'spark'

bi <- tidy_desc %>%
  filter(word == "bi") 
bi['skill'] = 'power bi'

aws1 <- tidy_desc %>%
  filter(word == "aws")
aws2 <- trigrams_united %>% 
  filter(str_detect(trigram, "amazon web service")) %>% 
  rename(word = trigram)
aws <- rbind(aws1, aws2)
aws['skill'] = 'aws'

tool_skills <- rbind(tableau, excel, microsoft, hadoop, spark, bi, aws)
tool_skills <- tool_skills %>% 
  select(desc_number, skill)
tool_skills <- tool_skills[!duplicated(tool_skills), ]

percent_tool <- tool_skills %>% 
  group_by(skill) %>% 
  summarise(n = n())  %>% 
  mutate(proportion = n / nrow(jobs_data)) 
```
  
For tools it looks like Microsoft is really important, with Excel being the most desired.

```r
percent_tool %>% 
  ggplot(aes(x = skill, y = proportion)) +
  geom_bar(stat = "identity", fill = "#35785f") +
  labs(title = "Most Desired Tools") +
  coord_flip()
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-27-1.png" width="672" />
  
# Estimated Salary
Since I live in California I wanted to look at the min, mid, and max salary for each position.

```r
# overall
jobs_data %>% 
  filter(state == "CA") %>% 
  select(min, mid, max) %>% 
  summarise(mean_min = mean(min), mean_mid = mean(mid), mean_max = mean(max))
```

```
##   mean_min mean_mid mean_max
## 1 85.53769 111.6276 137.7175
```
  
As we can see data scientist has the higher mean salary overall, while data analyst has the lowest.

```r
# California
jobs_data %>% 
  filter(state == "CA") %>% 
  select(min, mid, max, job_type) %>% 
  group_by(job_type) %>% 
  summarise(mean_min = mean(min), mean_mid = mean(mid), mean_max = mean(max))
```

```
## # A tibble: 4 x 4
##   job_type         mean_min mean_mid mean_max
##   <chr>               <dbl>    <dbl>    <dbl>
## 1 business analyst     69.3     93.7     118.
## 2 data analyst         66.3     88.4     111.
## 3 data engineer        99.9    128.      155.
## 4 data scientist      104.     133.      162.
```
  
I had a lot of fun with this project it helped show me that I need to up my Python and SQL skills. After this project I feel in love with NLP, so expect to see more in the future.


