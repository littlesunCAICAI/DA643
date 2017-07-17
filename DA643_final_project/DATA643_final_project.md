DATA643 Final Project: Yelp Recommendation System
================
Yun Mai, Kelly Shaffer
July 16, 2017

Restaurant Recommendation System based on Yelp data
===================================================

### 1.Introduction

As modern consumers, we greatly benefit from restaurant recommendation applications. It is so convenient to get a list of restaurants that match our preferences without much clicking, comparing, and browsing through a long list of reviews for each single business.

In this project, we want to apply the algorithms to develop predictive models learned from the DATA643 course "of "Current Topic of Data Science - Recommendation System"" to build a restaurant recommendation system that suggests the most suitable restaurant for users. If time permits, we will build an application.

### 2.Motivation

It is very common that we hang out with families, friends, and coworkers when comes to lunch or dinner time. As the users of recommendation applications, we care more about how we will like a restaurant. We will tend to have happier experiences when the prediction of the recommendation system is as good as what it says. As there is a completed and big data set of user and restaurants reviews, we want to see whether we can use the latest techniques to make good predictions. In the data set, there are not only reviews but also relevant information of users and restaurants that allow us to do more complicated computation, which might lead to the construction of a better model.

### 3.Dataset

In this project, we will use a Yelp Dataset Challenge round 9 from yelp website. The dataset has 4.1M reviews and 947K tips by 1M users for 144K businesses; 1.1M business attributes, e.g. hours, parking availability, ambience; and aggregated check-ins over time for each of the 125K businesses. The data includes diverse sets of cities: Edinburgh in U.K.; Karlsruhe in Germany; Montreal and Waterloo in Canada; Pittsburgh, Charlotte, Urbana-Champaign, Phoenix, Las Vagas, Madison, and Cleveland in U.S.

``` r
install.packages("jsonlite",repos='http://cran.us.r-project.org')
devtools::install_github("sailthru/tidyjson")
install.packages("doParallel")
install.packages(('BBmisc'))
```

Load packages

``` r
suppressWarnings(suppressMessages(library(jsonlite)))
suppressWarnings(suppressMessages(library(tidyjson)))
suppressWarnings(suppressMessages(library(plyr)))
suppressWarnings(suppressMessages(library(dplyr)))
suppressWarnings(suppressMessages(library(recommenderlab)))
suppressWarnings(suppressMessages(library(knitr)))
suppressWarnings(suppressMessages(library(tidyr)))
suppressWarnings(suppressMessages(library(ggplot2)))

# user-item matrix
suppressWarnings(suppressMessages(library(stringi)))
suppressWarnings(suppressMessages(library(Matrix)))
```

#### 3.1 Load the pre-processed data

#### 3.1 Explore the data

Load the pre-processed data

``` r
# read data from Github repository
business<- read.csv("https://raw.githubusercontent.com/YunMai-SPS/DA643/master/DA643_final_project/business.csv")

user <- read.csv("https://raw.githubusercontent.com/YunMai-SPS/DA643/master/DA643_final_project/user_1.csv")
for (i in c(2:4)){
  a<- paste0(cat('"'),'https://raw.githubusercontent.com/YunMai-SPS/DA643/master/DA643_final_project/user_',i,'.csv',cat('"'))
  user_1 <- read.csv(a)
  user <- rbind(user, user_1)
}
```

    ## """"""

``` r
rating <- read.csv("https://raw.githubusercontent.com/YunMai-SPS/DA643/master/DA643_final_project/rating_1.csv")

for (i in c(2:7)){
  a<- paste0(cat('"'),'https://raw.githubusercontent.com/YunMai-SPS/DA643/master/DA643_final_project/rating_',i,'.csv',cat('"'))
  rating_1 <- read.csv(a)
  rating <- rbind(rating, rating_1)
}
```

    ## """"""""""""

``` r
rating_copy <- rating
```

\*\* View the data\*\*

``` r
#rearrange the column
rating <- rating[,c("restaurant", "business_id", "user", "user_id","stars", "useful", "funny", "cool" ,"document.id")]

kable(head(rating,n=5))
```

| restaurant                                 | business\_id            | user   | user\_id               |  stars|  useful|  funny|  cool|  document.id|
|:-------------------------------------------|:------------------------|:-------|:-----------------------|------:|-------:|------:|-----:|------------:|
| Daily Kitchen Modern Eatery and Rotisserie | YCEZLECK9IToE8Mysorbhw  | Monera | ---1lKK3aKOuomHnwAkAow |      5|       3|      0|     2|        54219|
| The Placenta Lady                          | D1PhUlkQA1ZsVe9Cx4yqOw  | Monera | ---1lKK3aKOuomHnwAkAow |      5|       1|      1|     0|        14186|
| Fresh Mama                                 | 5aeR9KcboZmhDZlFscnYRA  | Monera | ---1lKK3aKOuomHnwAkAow |      5|       1|      0|     0|         3864|
| Red Velvet Cafe                            | t6WY1IrohUecqNjd9bG42Q  | Monera | ---1lKK3aKOuomHnwAkAow |      4|       2|      0|     0|        51335|
| Echo & Rig                                 | igHYkXZMLAc9UdV5VnR\_AA | Monera | ---1lKK3aKOuomHnwAkAow |      5|       0|      0|     0|         3774|

``` r
# convert ratings data to realRatingMatrix for implement of recommenderlab package

#check how many user and how many restaurant we have 
#length(unique(rating[,"user"]))  [1] 63081
#length(unique(rating[,"restaurant"])) [1] 65432


#builkd the user-item matrix
udf <- data.frame(user_No= seq(63081),user= unique(rating[,"user"]))
idf <- data.frame(restaurant_No= seq(65432),restaurant=unique(rating[,"restaurant"]))

rating <- merge(rating,udf,by.x='user',by.y='user')
rating <- merge(rating,idf,by.x='restaurant',by.y='restaurant')

rating_mx <- sparseMatrix(
  i =  rating$user_No, 
  j =  rating$restaurant_No, 
  x = rating$stars, 
  dimnames = list(levels(rating$user_No), levels(rating$restaurant_No))
)

mx <- as(rating_mx,"realRatingMatrix")

#Normalize by subtracting the row mean from all ratings in the row
mx_n <- normalize(mx)

image(mx, main = "Yelp restarurant reviews Data")
```

![](DATA643_final_project_files/figure-markdown_github-ascii_identifiers/unnamed-chunk-6-1.png)

``` r
image(mx_n, main = "Normalized Yelp restarurant reviews Data")
```

![](DATA643_final_project_files/figure-markdown_github-ascii_identifiers/unnamed-chunk-6-2.png)

\*\* Statistics of ratings data\*\*

``` r
# use visualize_ratings function from SVDApproximation to visualize statistics for all ratings: item count of different ratings,item histogram of users' average ratings, item histogram of items' average ratings, item histogram of number of rated items by user, item histogram of number of scores items have

#distribution of ratings
rating_frq <- as.data.frame(table(rating$stars))

ggplot(rating_frq,aes(Var1,Freq)) +   
  geom_bar(aes(fill = Var1), position = "dodge", stat="identity",fill="palegreen")+ labs(x = "Stars")
```

![](DATA643_final_project_files/figure-markdown_github-ascii_identifiers/unnamed-chunk-7-1.png)

``` r
#calculate average reviews for each restaurant
business_mean <- data.frame(restaurant = idf$restaurant, average_stars=colMeans(mx))

par(mfrow=c(2,2))

ggplot(user,aes(review_count)) +
  geom_histogram(binwidth = 0.05,col='red',fill="plum") + coord_cartesian(ylim=c(0,12000)) + labs(x = "User Average Reviews")+geom_vline(xintercept = mean(user$review_count),col='blue',size=1)
```

![](DATA643_final_project_files/figure-markdown_github-ascii_identifiers/unnamed-chunk-7-2.png)

``` r
ggplot(business,aes(review_count)) +
  geom_histogram(binwidth = 0.05,col='blue',fill="sandybrown") + coord_cartesian(ylim=c(0,7000)) + labs(x = "Restaurant Average Reviews")+geom_vline(xintercept = mean(business$review_count),col='red',size=1)
```

![](DATA643_final_project_files/figure-markdown_github-ascii_identifiers/unnamed-chunk-7-3.png)

``` r
ggplot(user,aes(average_stars)) +
  geom_histogram(binwidth = 0.03,fill="plum")  + labs(x = "Mean of Reviews User Gives")
```

![](DATA643_final_project_files/figure-markdown_github-ascii_identifiers/unnamed-chunk-7-4.png)

``` r
ggplot(business_mean,aes(average_stars)) +
  geom_histogram(binwidth = 0.03,fill="sandybrown") + labs(x = "Mean of Reviews Restaurant Has")
```

![](DATA643_final_project_files/figure-markdown_github-ascii_identifiers/unnamed-chunk-7-5.png)

### Algorithms

In this project, we will use a Yelp Dataset Challenge round 9 from yelp website. The dataset has 4.1M reviews and 947K tips by 1M users for 144K businesses; 1.1M business attributes, e.g. hours, parking availability, ambience; and aggregated check-ins over time for each of the 125K businesses. The data includes diverse sets of cities: Edinburgh in U.K.; Karlsruhe in Germany; Montreal and Waterloo in Canada; Pittsburgh, Charlotte, Urbana-Champaign, Phoenix, Las Vagas, Madison, and Cleveland in U.S.

In the Yelp dataset there is more information other than only ratings, so we can not only use content-based algorithm but also collaborative filtering algorithms. Location of the restaurant is an important factor to do the recommendation, so the location will be considered so the similarity between the distance and similarity between user/items will be combined. Other algorithms like alternative linear squares and singular value decomposition will also be used to build the prediction models.

Because there are three criteria in reviews: funny, useful, and cool, the rating will be calculated as follows:

*R* : *U**s**e**r**s* × *I**t**e**m**s* → *R*<sub>0</sub> × *R*<sub>1</sub> × ...*R*<sub>*k*</sub>

*R*<sub>0</sub> is the set of possible overall rating values, and *R*<sub>*i*</sub> represents the possible rating values for each individual criterion i (i = 1,..,k), typically on some numeric scale.

The prediction results of single-criteria collaborative filtering algorithm and multi-criteria collaborative filtering algorithms will be compared to decide which approach is better.

The implementation and evaluation will be performed in R and Apache Spark. At last, if time permits, an application will be built with the Shiny package.

### Reference:

1.  Blanca Vargas-Govea, Gabriel González-Serna, Rafael Ponce-Medellín. Effects of relevant contextual features in the performance of a restaurant recommender system.CARS,( 2011)
2.  Mengqi Yu, Meng Xue, Wenjia Ouyang. Restaurants Review Star Prediction for Yelp Dataset.Conference Proceedings (2015).
3.  Gediminas Adomavicius, YoungOk Kwon. New Recommendation Techniques for Multi-Criteria Rating Systems. IEEE Intelligent Systems 22-3 (2017).
4.  Jun Zeng, Feng Li, Haiyang Liu, Junhao Wen, Sachio Hirokawa. A Restaurant Recommender System Based on User Preference and Location in Mobile Environment. Advanced Applied Informatics (IIAI-AAI), 2016 5th IIAI International Congress.
