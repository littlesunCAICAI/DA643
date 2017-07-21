DATA643 Final Project: Restaurant Recommendation System
================
Yun Mai, Kelly Shaffer
July 16, 2017

Restaurant Recommendation System based on Yelp data
===================================================

### 1. Introduction

As modern consumers, we greatly benefit from restaurant recommendation applications. It is so convenient to get a list of restaurants that match our preferences without much clicking, comparing, and browsing through a long list of reviews for each single business.

In this project, we want to apply the algorithms to develop predictive models learned from the DATA643 course "of"Current Topic of Data Science - Recommendation System"" to build a restaurant recommendation system that suggests the most suitable restaurant for users.

### 2. Motivation

It is very common that we hang out with families, friends, and coworkers when comes to lunch or dinner time. As the users of recommendation applications, people care more about how we will like a restaurant. People will tend to have happier experiences when the prediction of the recommendation system is as good as what it says. As there is a completed and big data set of user and restaurants reviews, we want to see whether we can use the latest techniques to make good predictions. In the data set, there are not only reviews but also relevant information of users and restaurants that allow us to do more complicated computation, which might lead to the construction of a better model.

### 3. Aim

3.1 In this project, we will use collaborative filtering algorithms to build the primary recommendation system.

3.2 Location of the restaurant is an important factor to be considered when building a restaurant recommendation system. The location will be used to filter the restaurants from a top50 list.

3.3 In the Yelp dataset there is more information other than only ratings. There are three criteria in reviews: funny, useful, and cool and these factors will be integrated to the primary ratings. We hope to increase the diversity and serendipity of the results of the recommendation system.

### 4. Dataset

In this project, we will use a Yelp Dataset Challenge round 9 from Yelp website. The dataset has 4.1M reviews and 947K tips by 1M users for 144K businesses; 1.1M business attributes, such as hours, parking availability, ambiance; and aggregated check-ins over time for each of the 125K businesses. The data includes diverse sets of cities: Edinburgh in U.K.; Karlsruhe in Germany; Montreal and Waterloo in Canada; Pittsburgh, Charlotte, Urbana-Champaign, Phoenix, Las Vagas, Madison, and Cleveland in U.S.

``` r
install.packages("jsonlite",repos='http://cran.us.r-project.org')
devtools::install_github("sailthru/tidyjson")
install.packages("doParallel")
install.packages(('BBmisc'))
install.packages("DT")  
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
suppressWarnings(suppressMessages(library(DT)))
```

#### 4.1 Process the raw data

#### 4.2 Explore the data

**Load the pre-processed data**

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
# save a copy  
rating_copy  <- rating
```

**View the data**

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

# length(unique(rating[,"user"]))  [1] 63081
# length(unique(rating[,"restaurant"])) [1] 65432

       
#build the user-item matrix
udf <- data.frame(user_No= seq(1:length(unique(rating[,"user"]))),user= unique(rating[,"user"]))
idf <- data.frame(restaurant_No= seq(1:length(unique(rating[,"restaurant"]))),restaurant=unique(rating[,"restaurant"]))

rating <- merge(rating,udf,by.x='user',by.y='user')
rating <- merge(rating,idf,by.x='restaurant',by.y='restaurant')

rating_mx <- sparseMatrix(
  i =  rating$user_No, 
  j =  rating$restaurant_No, 
  x = rating$stars, 
  dimnames = list(levels(rating$user_No), levels(rating$restaurant_No))
)

#converting dcGMatrix to realRatingMatrix for applyting recommenderlab
mx <- as(rating_mx,"realRatingMatrix")
#setting itemlabels
colnames(mx) <- paste("R", 1:65432, sep = "")
as(mx[1,1:10],"list")
## [[1]]
##  R1  R2  R3  R4  R5  R6  R7  R8  R9 R10 
##   5   5   5   4   5   5   5   5   5   4

#setting userlabels
rownames(mx) <- paste("U", 1:63081, sep = "")
as(mx[1,1:10], "list")
## $U1
##  R1  R2  R3  R4  R5  R6  R7  R8  R9 R10 
##   5   5   5   4   5   5   5   5   5   4

#Normalize by subtracting the row mean from all ratings in the row
mx_n <- normalize(mx)

#view the matrix
getRatingMatrix(mx)[1:10,1:5]
## 10 x 5 sparse Matrix of class "dgCMatrix"
##     R1 R2 R3 R4 R5
## U1   5  5  5  4  5
## U2   .  .  .  .  5
## U3   .  .  .  .  .
## U4   .  .  .  .  .
## U5   .  .  .  .  .
## U6   .  .  .  1  5
## U7   .  .  .  4  5
## U8   1  .  .  .  5
## U9   .  .  .  .  .
## U10  .  .  .  4  .

image(mx, main = "Yelp restarurant reviews Data")
```

![](DATA643_final_project_files/figure-markdown_github-ascii_identifiers/unnamed-chunk-6-1.png)

``` r
image(mx_n, main = "Normalized Yelp restarurant reviews Data")
```

![](DATA643_final_project_files/figure-markdown_github-ascii_identifiers/unnamed-chunk-6-2.png)

**Statistics of ratings data**

``` r
# use visualize_ratings function from SVDApproximation to visualize statistics for all ratings: item count of different ratings,item histogram of users' average ratings, item histogram of items' average ratings, item histogram of number of rated items by user, item histogram of number of scores items have

summary(rating[, 'stars'])
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##   1.000   3.000   4.000   3.716   5.000   5.000

``` r
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
  geom_histogram(binwidth = 0.05,col='red',fill="plum") + coord_cartesian(ylim=c(0,12000)) + labs(x = "User Review COunt")+geom_vline(xintercept = mean(user$review_count),col='blue',size=1)
```

![](DATA643_final_project_files/figure-markdown_github-ascii_identifiers/unnamed-chunk-7-2.png)

``` r
ggplot(business,aes(review_count)) +
  geom_histogram(binwidth = 0.05,col='blue',fill="sandybrown") + coord_cartesian(ylim=c(0,7000)) + labs(x = "Restaurant Review COunt")+geom_vline(xintercept = mean(business$review_count),col='red',size=1)
```

![](DATA643_final_project_files/figure-markdown_github-ascii_identifiers/unnamed-chunk-7-3.png)

``` r
ggplot(user,aes(average_stars)) +
  geom_histogram(binwidth = 0.03,fill="plum")  + labs(x = "User Average Review")
```

![](DATA643_final_project_files/figure-markdown_github-ascii_identifiers/unnamed-chunk-7-4.png)

``` r
ggplot(business_mean,aes(average_stars)) +
  geom_histogram(binwidth = 0.03,fill="sandybrown") + labs(x = "Restaurant Average Review")
```

![](DATA643_final_project_files/figure-markdown_github-ascii_identifiers/unnamed-chunk-7-5.png)

``` r
round_r <- sum(user$average_stars == 1)+sum(user$average_stars == 2)+sum(user$average_stars == 3)+sum(user$average_stars == 4)+sum(user$average_stars == 5)
print(paste("Total number of people who had rounded average ratings:",round_r))
## [1] "Total number of people who had rounded average ratings: 405551"
user_rate_1 <- sum(user$review_count == 1)
user_rate_2 <- sum(user$review_count == 2)
user_rate_3 <- sum(user$review_count == 3)
user_rate_4 <- sum(user$review_count == 4)
print(paste("Number of people who only rated one restaurant:",user_rate_1))
## [1] "Number of people who only rated one restaurant: 189809"
print(paste("Number of people who only rated twice:",user_rate_2))
## [1] "Number of people who only rated twice: 126347"
print(paste("Number of people who only rated three times:",user_rate_3))
## [1] "Number of people who only rated three times: 96815"
print(paste("Number of people who only rated four times:",user_rate_4))
## [1] "Number of people who only rated four times: 69627"
print(paste("Number of people who only rated less than three times:",user_rate_1 + user_rate_2 +user_rate_3))
## [1] "Number of people who only rated less than three times: 412971"
```

**From raw data we can see:**

**1.** Rating distribution is not normal with the most frequent rating at the highest rating 5, whose frequency is much higher than other ratings. One possibility is that people who would write reviews for a restaurant on Yelp are those who will view review/ratings online before deciding to try a new restaurant. So there is more chance that these people like what they chose. This suggests that the current restaurant recommendation systems work very well so it is more likely that people could find the food they like by searching for the recommender engine/application.

**2.** Distribution of user review count is not normal with an average at 24. Majority people only wrote a few reviews and there are very few people wrote thousands of reviews with a maximum number at 11284.By looking at the minimum review count we knew that some people did not write any review.

**3.** Distribution of user review count is not normal with an average at 28. Majority restaurant received a few reviews and there is very few restaurant received thousands of reviews with a maximum number at 6414. If we look at the minimum review count, we can see any restaurant in this data set at least got 3 reviews.

**4.** The average rating for each user is multimodal distribution. The count of average rating at each round number(stars) is much higher than other not rounded number. In consistent with Figure 1, average rating at 5 has the highest frequency. The possible reasons that a lot of people had a rounded average rating could either be these people only give the same rating for a different restaurant and they only rated very few restaurants. It is interesting to notice that the number of people who had rounded average ratings, 405551, is close to the number of people who only rated less than three times, 412971.

**5.** Similar to the user average rating, the average rating for each restaurant is multimodal distribution. In consistent with Figure 1, average rating at 5 has the highest frequency. One of the possible reason for this pattern is there was a large number of restaurant received very few ratings and ratings were the same. Another reason is that there are a lot of very good restaurants always received 5. But, is it really possible?

### 5.Creating a Recommender

#### 5.1 Building a User-based Collaborative Filtering Model

``` r
# check if there is abnormal ratings in the data
table(mx@data@x[] > 5)
```

    ## 
    ##   FALSE    TRUE 
    ## 1409140       1

``` r
table(mx@data@x[] < 1)
```

    ## 
    ##   FALSE 
    ## 1409141

``` r
# set the abnormal rating to a most closed normal number
mx@data@x[mx@data@x[] > 5] <- 5

# Keeping only restaurants with more than 50 ratings and users with more than 20 rating
mx_r <- mx[rowCounts(mx) > 20,]
mx_r <- mx_r[,colCounts(mx_r) > 50]

# creating the evaluation scheme, separate the data into train set and test set
set.seed(1)
(e <- evaluationScheme(mx_r[1:1200], method = "split",train = 0.8, given = 5, goodRating = 3, k=5))
```

    ## Evaluation scheme with 5 items given
    ## Method: 'split' with 5 run(s).
    ## Training set proportion: 0.800
    ## Good ratings: >=3.000000
    ## Data set: 1200 x 5243 rating matrix of class 'realRatingMatrix' with 488248 ratings.

``` r
# Creating a user-based collaborative filtering model using the training data.
(r_ubcf <- Recommender(getData(e, "train"), method ="UBCF", parameter = list(method = "cosine", normalize = "Z-score", nn=25)))
```

    ## Recommender of type 'UBCF' for 'realRatingMatrix' 
    ## learned using 960 users.

``` r
# release memory
gc()
```

    ##            used  (Mb) gc trigger  (Mb)  max used  (Mb)
    ## Ncells  3714349 198.4   12002346 641.0  12002346 641.0
    ## Vcells 55754508 425.4  112780893 860.5 112769232 860.4

``` r
# Increasing the storage capacity
memory.limit(size=700000)
```

    ## [1] 7e+05

``` r
names(getModel(r_ubcf))
```

    ## [1] "description" "data"        "method"      "nn"          "sample"     
    ## [6] "normalize"   "verbose"

``` r
# evaluation  
results <- evaluate(e, method="UBCF", type = "ratings", n=c(1,3,5,10,15,20))
```

    ## UBCF run fold/sample [model time/prediction time]
    ##   1  [0.1sec/11.53sec] 
    ##   2  [0.05sec/11.86sec] 
    ##   3  [0.03sec/11.55sec] 
    ##   4  [0.04sec/11.29sec] 
    ##   5  [0.05sec/11.72sec]

``` r
avg(results)
```

    ##         RMSE      MSE      MAE
    ## res 1.485573 2.207031 1.221872

``` r
# making predictions on ratings
(p_rating <- predict(r_ubcf, getData(e, "known"), type="ratings",n=10))
```

    ## 240 x 5243 rating matrix of class 'realRatingMatrix' with 1246644 ratings.

``` r
# show predicted ratings
as(p_rating, "matrix")[1:10,1:7]
```

    ##           R1       R3       R4       R5       R6      R10      R11
    ## U11 4.377811 4.400000 4.448639 4.704822 4.400000 4.476386 4.335383
    ## U31 2.800000 2.802313 2.700283 3.000264 2.800000 2.800000 2.874564
    ## U40 3.895722 4.000000 3.981405 4.183772 4.147711 4.032290 4.304996
    ## U41 4.350739 4.399499 4.426552 4.561826 4.434757 4.400000 4.376478
    ## U43 3.800000 3.800000 3.757982 3.856515 3.830139 3.827601 3.888330
    ## U45 4.221984 4.200000 4.200000 4.320884 4.200000 4.200000 4.197166
    ## U46 2.400000 2.473057 2.219222 2.466187 2.400000 2.400000 2.473215
    ## U53 3.400000 3.400000 3.306760 3.672172 3.534933 3.436452 3.111497
    ## U54 3.614614 3.622771 3.760030 3.818249 3.600000 3.550045 3.445086
    ## U59 3.873293 3.800000 3.864313 4.149622 3.802304 3.874812 3.838606

``` r
# RMSE for n=10
error <- data.frame(calcPredictionAccuracy(p_rating, getData(e, "unknown")))
kable(error,caption="RMSE for n=10")
```

|      |  calcPredictionAccuracy.p\_rating..getData.e...unknown...|
|------|---------------------------------------------------------:|
| RMSE |                                                  1.468892|
| MSE  |                                                  2.157643|
| MAE  |                                                  1.188701|

``` r
# making predictions for User in the first row of test data set on topNList
(p_topN <- predict(r_ubcf, mx_r[1201],type="topNList",n=10))
```

    ## Recommendations as 'topNList' with n = 10 for 1 users.

``` r
# show predicted top10 restaurants
(pri_rec <- as(p_topN, "list"))
```

    ## $U1827
    ##  [1] "R1030" "R478"  "R745"  "R1550" "R1344" "R6798" "R5179" "R228" 
    ##  [9] "R1204" "R229"

In practice, we have to consider the location while designing a restaurant recommendation system. In most of the time, people will use a recommendation engine to find a restaurant from a certain city.

``` r
#get city info from business data
city <- business[,c('name','city','state')]
city <- city[!duplicated(city$name),]
colnames(city) <- c('restaurant','city','state')
idf_city <- left_join(idf,city,by='restaurant')
```

    ## Warning: Column `restaurant` joining factors with different levels,
    ## coercing to character vector

``` r
idf_city$restaurant_id <- paste("R", 1:65432, sep = "")
idf_city$city <- as.character(idf_city$city)
idf_city$state <- as.character(idf_city$state)

#get 50 restaurants for User 1201 from recemmender system
(p_top100 <- predict(r_ubcf, mx_r[1201],type="topNList",n=50))
```

    ## Recommendations as 'topNList' with n = 50 for 1 users.

``` r
# filter the restaurant for User 1201 based on location
pred_restaurant <-  data.frame(as(p_top100, "list"))
colnames(pred_restaurant) <- "U1201"
pred_restaurant[] <- lapply(pred_restaurant, as.character)
pred_restaurant$restaurant_id <- pred_restaurant$U1201

pred_restaurant <- left_join(pred_restaurant,idf_city, by='restaurant_id' )
pred_restaurant$city <- as.character(pred_restaurant$city)
pred_restaurant$state <- as.character(pred_restaurant$state)

# For example, if user 1201 want to get recommendation for restaurants in Las vegas, we can find out from the top100 list
Lasvegas <- filter(pred_restaurant,city == "Las Vegas")
head(Lasvegas, n=5)
```

    ##   U1201 restaurant_id restaurant_No                    restaurant
    ## 1 R1030         R1030          1030 Desert Wireless iPhone Repair
    ## 2  R478          R478           478                    SkinnyFATS
    ## 3 R6798         R6798          6798                    9037 Salon
    ## 4 R5179         R5179          5179                    Lucki Thai
    ## 5  R228          R228           228                  Bachi Burger
    ##        city state
    ## 1 Las Vegas    NV
    ## 2 Las Vegas    NV
    ## 3 Las Vegas    NV
    ## 4 Las Vegas    NV
    ## 5 Las Vegas    NV

#### 5.2 Multi-Criteria Recommender System

Because there are three criteria in reviews: funny, useful, and cool, the rating will be calculated as follows:

*R* : *U**s**e**r**s* × *I**t**e**m**s* → *R*<sub>0</sub> × *R*<sub>1</sub> × ...*R*<sub>*k*</sub>

*R*<sub>0</sub> is the set of possible overall rating values, and *R*<sub>*i*</sub> represents the possible rating values for each individual criterion i (i = 1,..,k), typically on some numeric scale.

The prediction results of single-criteria collaborative filtering algorithm and multi-criteria collaborative filtering algorithms will be compared to decide which approach is better.

The implementation and evaluation will be performed in R and Apache Spark. At last, if time permits, an application will be built with the Shiny package.

#### 5.2.1 Building the User-item Matrix Based on Useful, Funny, and Cool Comments

**Useful Matrix**

``` r
#build the user-item matrix based on funny comments
useful_mx <- sparseMatrix(
  i =  rating$user_No, 
  j =  rating$restaurant_No, 
  x = rating$useful, 
  dimnames = list(levels(rating$user_No), levels(rating$restaurant_No))
)

#converting dcGMatrix to realRatingMatrix for applyting recommenderlab
u_mx <- as(useful_mx,"realRatingMatrix")

#setting itemlabels
colnames(u_mx) <- paste("R", 1:65432, sep = "")

#setting userlabels
rownames(u_mx) <- paste("U", 1:63081, sep = "")


#view the matrix
getRatingMatrix(u_mx)[1:10,1:5]
```

    ## 10 x 5 sparse Matrix of class "dgCMatrix"
    ##     R1 R2 R3 R4 R5
    ## U1   3  1  1  2  0
    ## U2   .  .  .  .  0
    ## U3   .  .  .  .  .
    ## U4   .  .  .  .  .
    ## U5   .  .  .  .  .
    ## U6   .  .  .  4  0
    ## U7   .  .  .  1  0
    ## U8   1  .  .  .  3
    ## U9   .  .  .  .  .
    ## U10  .  .  .  1  .

**Funny Matrix**

``` r
#build the user-item matrix based on funny comments
funny_mx <- sparseMatrix(
  i =  rating$user_No, 
  j =  rating$restaurant_No, 
  x = rating$funny, 
  dimnames = list(levels(rating$user_No), levels(rating$restaurant_No))
)

#converting dcGMatrix to realRatingMatrix for applyting recommenderlab
f_mx <- as(funny_mx,"realRatingMatrix")

#setting itemlabels
colnames(f_mx) <- paste("R", 1:65432, sep = "")

#setting userlabels
rownames(f_mx) <- paste("U", 1:63081, sep = "")

#view the matrix
getRatingMatrix(f_mx)[1:10,1:5]
```

    ## 10 x 5 sparse Matrix of class "dgCMatrix"
    ##     R1 R2 R3 R4 R5
    ## U1   0  1  0  0  0
    ## U2   .  .  .  .  0
    ## U3   .  .  .  .  .
    ## U4   .  .  .  .  .
    ## U5   .  .  .  .  .
    ## U6   .  .  .  0  0
    ## U7   .  .  .  0  0
    ## U8   0  .  .  .  0
    ## U9   .  .  .  .  .
    ## U10  .  .  .  0  .

**Cool Matrix**

``` r
#build the user-item matrix based on funny comments
cool_mx <- sparseMatrix(
  i =  rating$user_No, 
  j =  rating$restaurant_No, 
  x = rating$cool, 
  dimnames = list(levels(rating$user_No), levels(rating$restaurant_No))
)

#converting dcGMatrix to realRatingMatrix for applyting recommenderlab
c_mx <- as(cool_mx,"realRatingMatrix")

#setting itemlabels
colnames(c_mx) <- paste("R", 1:65432, sep = "")

#setting userlabels
rownames(c_mx) <- paste("U", 1:63081, sep = "")

#view the matrix
getRatingMatrix(c_mx)[1:10,1:5]
```

    ## 10 x 5 sparse Matrix of class "dgCMatrix"
    ##     R1 R2 R3 R4 R5
    ## U1   2  0  0  0  0
    ## U2   .  .  .  .  1
    ## U3   .  .  .  .  .
    ## U4   .  .  .  .  .
    ## U5   .  .  .  .  .
    ## U6   .  .  .  1  0
    ## U7   .  .  .  1  1
    ## U8   0  .  .  .  0
    ## U9   .  .  .  .  .
    ## U10  .  .  .  1  .

``` r
# statistic of useful, funny and cool comments data
summary(u_mx@data@x[])
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##   0.000   0.000   0.000   1.006   1.000 500.000

``` r
summary(f_mx@data@x[])
```

    ##     Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
    ##   0.0000   0.0000   0.0000   0.4091   0.0000 287.0000

``` r
summary(c_mx@data@x[])
```

    ##     Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
    ##   0.0000   0.0000   0.0000   0.5151   0.0000 234.0000

From the summary, we can see the values of useful, funny or cool represent how many people felt the reviews for the restaurant are useful, funny or cool. The higher the value, the more popular of these restaurants are. We can consider these number as ratings from different aspects. Because the scale of these three factors is different, as you can see from the max value was 500, 287, and 234, we will convert them to binary values. Then the ratings for useful, funny or cool will be combined with the primary ratings to build the new recommender models.

``` r
# the frequeny of restaurant's rating is useful 
useful_tb <- as.data.frame(table(rating$useful))
useful_tb$Var1 <- as.numeric(as.character(useful_tb$Var1))
# how many pepople believed that the review was useful at a threshold at 100 restaurants having the same amount of "useful" notes
u_threshold <- useful_tb[useful_tb$Freq > 50,]

# the frequeny of restaurant's rating is useful 
funny_tb <- as.data.frame(table(rating$funny))
funny_tb$Var1 <- as.numeric(as.character(funny_tb$Var1))
# how many pepople believed that the review was useful at a threshold at 100 restaurants having the same amount of "useful" notes
f_threshold <- funny_tb[funny_tb$Freq > 50,]

# the frequeny of restaurant's rating is useful 
cool_tb <- as.data.frame(table(rating$useful))
cool_tb$Var1 <- as.numeric(as.character(cool_tb$Var1))
# how many pepople believed that the review was useful at a threshold at 100 restaurants having the same amount of "useful" notes
c_threshold <- cool_tb[cool_tb$Freq > 50,]

mx_b <-mx_r
# convert the basic rating matrix to binary matrix
mx_b@data@x [mx_b@data@x < mean(mx_b@data@x[])]<- 1
mx_b@data@x [mx_b@data@x > mean(mx_b@data@x[])]<- 0

# convert the useful matrix to binary matrix
u_mx@data@x [u_mx@data@x < max(u_threshold$Var1)]<- 1
u_mx@data@x [u_mx@data@x > max(u_threshold$Var1)]<- 0

# convert the funny rating matrix to binary matrix  
f_mx@data@x [f_mx@data@x < max(f_threshold$Var1)]<- 1
f_mx@data@x [f_mx@data@x > max(f_threshold$Var1)]<- 0

# convert the cool rating matrix to binary matrix
c_mx@data@x [c_mx@data@x < max(c_threshold$Var1)]<- 1
c_mx@data@x [c_mx@data@x > max(c_threshold$Var1)]<- 0


#chose the users and restaurants matching the constrained user-item matrix which users rated the restaurant more than 20 times and restaurants received more than 50 reviews.
u_mx_fit <- u_mx[,c(colnames(mx_r))]
u_mx_fit <- u_mx_fit[row.names(u_mx_fit) %in% c(rownames(mx_r)),]

f_mx_fit <- f_mx[,c(colnames(mx_r))]
f_mx_fit <- f_mx_fit[row.names(f_mx_fit) %in% c(rownames(mx_r)),]

c_mx_fit <- c_mx[,c(colnames(mx_r))]
c_mx_fit <- c_mx_fit[row.names(c_mx_fit) %in% c(rownames(mx_r)),]

# combine primary ratings with useful rating by element-wise multiplication  
r0_r1 <- mx_b@data * u_mx_fit@data

# combine primary ratings with funny rating by element-wise multiplication  
r0_r1_r2 <- r0_r1 * f_mx_fit@data

# combine primary ratings with cool rating by element-wise multiplication  
r0_r1_r2_r3 <- r0_r1_r2 * c_mx_fit@data
```

#### 5.2.2 Building the Recommendation Systems Based on Multi\_criteria Rating

There are 7 ways to intergrate useful, funny, cool, and primary ratings: primary+useful,primary+funny, primary+cool, primary+useful+funny, primary+useful+cool, primary+cool+funny, primary+useful+funny+cool. We will use primary+useful,primary+useful+funny, and primary+useful+funny+cool to build the recommendation models.

\*\* Primary + Useful\*\*

``` r
combine_1 <- as(r0_r1,"realRatingMatrix")

# creating the evaluation scheme, separate the data into train set and test set
set.seed(2)
(c1_e <- evaluationScheme(combine_1[1:1200], method = "split",train = 0.8, given = 5, goodRating = 3, k=5))
```

    ## Evaluation scheme with 5 items given
    ## Method: 'split' with 5 run(s).
    ## Training set proportion: 0.800
    ## Good ratings: >=3.000000
    ## Data set: 1200 x 5243 rating matrix of class 'realRatingMatrix' with 488248 ratings.

``` r
# Creating a user-based collaborative filtering) using the training data.
(c1_ubcf <- Recommender(getData(c1_e, "train"), method ="UBCF", parameter = list(method = "cosine", normalize = "Z-score", nn=25)))
```

    ## Recommender of type 'UBCF' for 'realRatingMatrix' 
    ## learned using 960 users.

``` r
# release memory
gc()
```

    ##            used  (Mb) gc trigger  (Mb)  max used  (Mb)
    ## Ncells  3736254 199.6   12002346 641.0  12002346 641.0
    ## Vcells 78251287 597.1  112780893 860.5 112780195 860.5

``` r
# evaluation  
c1_results <- evaluate(c1_e, method="UBCF", type = "ratings", n=c(1,3,5,10,15,20))
```

    ## UBCF run fold/sample [model time/prediction time]
    ##   1  [0.03sec/11.39sec] 
    ##   2  [0.05sec/12.31sec] 
    ##   3  [0.06sec/11.58sec] 
    ##   4  [0.03sec/11.68sec] 
    ##   5  [0.05sec/12.15sec]

``` r
avg(results)
```

    ##         RMSE      MSE      MAE
    ## res 1.485573 2.207031 1.221872

``` r
# making predictions on ratings
(c1_p_rating <- predict(c1_ubcf, getData(c1_e, "known"), type="ratings",n=10))
```

    ## 240 x 5243 rating matrix of class 'realRatingMatrix' with 1068552 ratings.

``` r
# show predicted ratings
as(c1_p_rating, "matrix")[1:10,1:7]
```

    ##            R1        R3        R4        R5        R6       R10       R11
    ## U6         NA        NA        NA        NA        NA        NA        NA
    ## U14 0.4525209 0.4328711 0.4502401 0.3393765 0.3895817 0.3967006 0.3460426
    ## U30 0.6000000 0.6000000 0.6000000 0.5603706 0.6000000 0.5865870 0.6355804
    ## U33 0.1702462 0.2000000 0.2188049 0.2156242 0.2000000 0.2227773 0.2068239
    ## U40 0.3915823 0.3798296 0.3860204 0.3717654 0.4000000 0.3760555 0.3720130
    ## U41 0.4138108 0.4138108 0.4138108 0.3542383 0.4000000 0.3771508 0.3928913
    ## U44        NA        NA        NA        NA        NA        NA        NA
    ## U57 0.8000000 0.7912793 0.8094813 0.7847997 0.8000000 0.7742290 0.8069384
    ## U58 0.8000000 0.8220852 0.7909140 0.7721165 0.8000000 0.8000000 0.7839648
    ## U66 0.6000000 0.5872285 0.5824837 0.5836675 0.6000000 0.5885307 0.5629996

``` r
# RMSE
(error <- data.frame(calcPredictionAccuracy(c1_p_rating, getData(c1_e, "unknown"))))
```

    ##      calcPredictionAccuracy.c1_p_rating..getData.c1_e...unknown...
    ## RMSE                                                     0.5033015
    ## MSE                                                      0.2533124
    ## MAE                                                      0.4509785

``` r
# evaluation  
#(It took long time to run evaluate results of the command is put here)
#results <- evaluate(e, method="UBCF", type = "topNList", n=c(1,3,5,10,15,20))
#UBCF run fold/sample [model time/prediction time]
     #1  [0.16sec/398.42sec] 
     #2  [0.17sec/393.06sec] 
     #3  [0.27sec/391.93sec] 
     #4  [0.09sec/393.77sec] 
     #5  [0.16sec/395.01sec] 
# making predictions on topNList
(c1_p_topN <- predict(c1_ubcf, combine_1[1201],type="topNList",n=10))
```

    ## Recommendations as 'topNList' with n = 10 for 1 users.

``` r
# show predicted top10 restaurants
(c1_rec <- as(c1_p_topN, "list"))
```

    ## $U1827
    ##  [1] "R1967" "R831"  "R603"  "R1861" "R1971" "R873"  "R5580" "R294" 
    ##  [9] "R1622" "R5977"

**Primary + Useful + Funny**

``` r
combine_2 <- as(r0_r1_r2,"realRatingMatrix")

# creating the evaluation scheme, separate the data into train set and test set
set.seed(3)
(c2_e <- evaluationScheme(combine_2[1:1200], method = "split",train = 0.8, given = 5, goodRating = 3, k=5))
```

    ## Evaluation scheme with 5 items given
    ## Method: 'split' with 5 run(s).
    ## Training set proportion: 0.800
    ## Good ratings: >=3.000000
    ## Data set: 1200 x 5243 rating matrix of class 'realRatingMatrix' with 488248 ratings.

``` r
# Creating a user-based collaborative filtering) using the training data.
(c2_ubcf <- Recommender(getData(c2_e, "train"), method ="UBCF", parameter = list(method = "cosine", normalize = "Z-score", nn=25)))
```

    ## Recommender of type 'UBCF' for 'realRatingMatrix' 
    ## learned using 960 users.

``` r
# release memory
gc()
```

    ##            used  (Mb) gc trigger   (Mb)  max used   (Mb)
    ## Ncells  3736637 199.6   12002346  641.0  12002346  641.0
    ## Vcells 81954207 625.3  135417071 1033.2 135416434 1033.2

``` r
# evaluation  
c2_results <- evaluate(c2_e, method="UBCF", type = "ratings", n=c(1,3,5,10,15,20))
```

    ## UBCF run fold/sample [model time/prediction time]
    ##   1  [0.04sec/11.51sec] 
    ##   2  [0.05sec/11.79sec] 
    ##   3  [0.04sec/11.61sec] 
    ##   4  [0.03sec/11.9sec] 
    ##   5  [0.04sec/12.04sec]

``` r
avg(results)
```

    ##         RMSE      MSE      MAE
    ## res 1.485573 2.207031 1.221872

``` r
# making predictions on ratings
(c2_p_rating <- predict(c2_ubcf, getData(c2_e, "known"), type="ratings",n=10))
```

    ## 240 x 5243 rating matrix of class 'realRatingMatrix' with 1068552 ratings.

``` r
# show predicted ratings
as(c2_p_rating, "matrix")[1:10,1:7]
```

    ##            R1        R3        R4        R5        R6       R10       R11
    ## U8  0.4242750 0.3862496 0.3882118 0.3499197 0.3871293 0.4219942 0.3580959
    ## U24 0.6000000 0.6000000 0.5808039 0.5543080 0.6000000 0.6000000 0.5769046
    ## U26 0.6000000 0.6000000 0.6000000 0.5680516 0.6000000 0.6000000 0.5756454
    ## U29 0.2320670 0.1787733 0.1581266 0.1752761 0.1845924 0.1998294 0.2152066
    ## U36 0.4000000 0.4000000 0.3894468 0.3783530 0.4000000 0.4000000 0.4097481
    ## U37 0.2017338 0.1858843 0.2024211 0.2162648 0.1776786 0.2205786 0.1484986
    ## U55 0.4000000 0.4000000 0.3874557 0.3344267 0.4000000 0.4000000 0.3821346
    ## U65 0.4000000 0.4000000 0.4427417 0.3727352 0.4230476 0.4356060 0.3717501
    ## U71 0.4000000 0.4000000 0.4278132 0.3971104 0.4000000 0.3751178 0.4000000
    ## U75 0.4239906 0.4136335 0.4000000 0.3648116 0.3872801 0.3900600 0.4315422

``` r
# RMSE
(error <- data.frame(calcPredictionAccuracy(c2_p_rating, getData(c2_e, "unknown"))))
```

    ##      calcPredictionAccuracy.c2_p_rating..getData.c2_e...unknown...
    ## RMSE                                                     0.5203734
    ## MSE                                                      0.2707885
    ## MAE                                                      0.4677258

``` r
# evaluation  
#(It took long time to run evaluate results of the command is put here)
#results <- evaluate(e, method="UBCF", type = "topNList", n=c(1,3,5,10,15,20))
#UBCF run fold/sample [model time/prediction time]
     #1  [0.16sec/398.42sec] 
     #2  [0.17sec/393.06sec] 
     #3  [0.27sec/391.93sec] 
     #4  [0.09sec/393.77sec] 
     #5  [0.16sec/395.01sec] 
# making predictions on topNList
(c2_p_topN <- predict(c2_ubcf, combine_2[1201],type="topNList",n=10))
```

    ## Recommendations as 'topNList' with n = 10 for 1 users.

``` r
# show predicted top10 restaurants
(c2_rec <- as(c2_p_topN, "list"))
```

    ## $U1827
    ##  [1] "R1967" "R603"  "R602"  "R1081" "R831"  "R2291" "R1861" "R3438"
    ##  [9] "R873"  "R1464"

**Primary + Useful + Funny + Cool**

``` r
combine_3 <- as(r0_r1_r2_r3,"realRatingMatrix")

# creating the evaluation scheme, separate the data into train set and test set
set.seed(4)
(c3_e <- evaluationScheme(combine_3[1:1200], method = "split",train = 0.8, given = 5, goodRating = 3, k=5))
```

    ## Evaluation scheme with 5 items given
    ## Method: 'split' with 5 run(s).
    ## Training set proportion: 0.800
    ## Good ratings: >=3.000000
    ## Data set: 1200 x 5243 rating matrix of class 'realRatingMatrix' with 488248 ratings.

``` r
# Creating a user-based collaborative filtering) using the training data.
(c3_ubcf <- Recommender(getData(c3_e, "train"), method ="UBCF", parameter = list(method = "cosine", normalize = "Z-score", nn=25)))
```

    ## Recommender of type 'UBCF' for 'realRatingMatrix' 
    ## learned using 960 users.

``` r
# release memory
gc()
```

    ##            used  (Mb) gc trigger   (Mb)  max used   (Mb)
    ## Ncells  3736922 199.6   12002346  641.0  12002346  641.0
    ## Vcells 85647595 653.5  135417071 1033.2 135416826 1033.2

``` r
# evaluation  
c3_results <- evaluate(c3_e, method="UBCF", type = "ratings", n=c(1,3,5,10,15,20))
```

    ## UBCF run fold/sample [model time/prediction time]
    ##   1  [0.05sec/11.27sec] 
    ##   2  [0.03sec/11.54sec] 
    ##   3  [0.02sec/11.58sec] 
    ##   4  [0.05sec/11.86sec] 
    ##   5  [0.04sec/11.65sec]

``` r
avg(results)
```

    ##         RMSE      MSE      MAE
    ## res 1.485573 2.207031 1.221872

``` r
# making predictions on ratings
(c3_p_rating <- predict(c3_ubcf, getData(c3_e, "known"), type="ratings",n=10))
```

    ## 240 x 5243 rating matrix of class 'realRatingMatrix' with 1073790 ratings.

``` r
# show predicted ratings
as(c3_p_rating, "matrix")[1:10,1:7]
```

    ##            R1        R3        R4        R5        R6       R10       R11
    ## U5         NA        NA        NA        NA        NA        NA        NA
    ## U15 0.6000000 0.6000000 0.6135017 0.6118739 0.6000000 0.5788825 0.6000000
    ## U24 0.4073642 0.4305689 0.4000000 0.3906037 0.4000000 0.4053768 0.3680601
    ## U26 0.6000000 0.6000000 0.6000000 0.5651377 0.6000000 0.5697705 0.5912079
    ## U34 0.4000000 0.3854438 0.3843342 0.3705854 0.4000000 0.4000000 0.3843342
    ## U36 0.6000000 0.5840123 0.6271648 0.5732234 0.5868789 0.5868789 0.5964009
    ## U46 0.4000000 0.4000000 0.3791149 0.4038671 0.3888094 0.3892605 0.3790034
    ## U58 0.2136111 0.1774667 0.2443033 0.1718200 0.1867005 0.2313888 0.2165374
    ## U59 0.2000000 0.1908956 0.2000000 0.2164272 0.2000000 0.2000000 0.2334704
    ## U71 0.3776172 0.3776605 0.4154288 0.3869664 0.3878530 0.3710440 0.3925566

``` r
# RMSE
(error <- data.frame(calcPredictionAccuracy(c3_p_rating, getData(c3_e, "unknown"))))
```

    ##      calcPredictionAccuracy.c3_p_rating..getData.c3_e...unknown...
    ## RMSE                                                     0.5295649
    ## MSE                                                      0.2804390
    ## MAE                                                      0.4686014

``` r
# evaluation  
#(It took long time to run evaluate results of the command is put here)
#results <- evaluate(e, method="UBCF", type = "topNList", n=c(1,3,5,10,15,20))
#UBCF run fold/sample [model time/prediction time]
     #1  [0.16sec/398.42sec] 
     #2  [0.17sec/393.06sec] 
     #3  [0.27sec/391.93sec] 
     #4  [0.09sec/393.77sec] 
     #5  [0.16sec/395.01sec] 
# making predictions on topNList
(c3_p_topN <- predict(c3_ubcf, combine_3[1201],type="topNList",n=10))
```

    ## Recommendations as 'topNList' with n = 10 for 1 users.

``` r
# show predicted top10 restaurants
(c3_rec <- as(c3_p_topN, "list"))
```

    ## $U1827
    ##  [1] "R831"  "R603"  "R1861" "R1622" "R5580" "R63"   "R294"  "R6589"
    ##  [9] "R1464" "R2410"

``` r
#get 50 restaurants for User 1201 from recemmender system
(c1_p_top100 <- predict(c1_ubcf, mx_r[1201],type="topNList",n=50))
```

    ## Recommendations as 'topNList' with n = 50 for 1 users.

``` r
# filter the restaurant for User 1201 based on location
c1_pred_restaurant <-  data.frame(as(c1_p_top100, "list"))
colnames(c1_pred_restaurant) <- "U1201"
c1_pred_restaurant[] <- lapply(c1_pred_restaurant, as.character)
c1_pred_restaurant$restaurant_id <- c1_pred_restaurant$U1201

c1_pred_restaurant <- left_join(c1_pred_restaurant,idf_city, by='restaurant_id' )
c1_pred_restaurant$city <- as.character(c1_pred_restaurant$city)
c1_pred_restaurant$state <- as.character(c1_pred_restaurant$state)

# For example, if user 1201 want to get recommendation for restaurants in Las vegas, we can find out from the top100 list
head(Lasvegas <- filter(c1_pred_restaurant,city == "Las Vegas"),n=5)
```

    ##   U1201 restaurant_id restaurant_No                       restaurant
    ## 1 R5580         R5580          5580                         Cafe Rio
    ## 2  R603          R603           603   Bayside Buffet at Mandalay Bay
    ## 3  R142          R142           142                    Serendipity 3
    ## 4 R1622         R1622          1622                              FIX
    ## 5   R63           R63            63 Luxor Hotel and Casino Las Vegas
    ##        city state
    ## 1 Las Vegas    NV
    ## 2 Las Vegas    NV
    ## 3 Las Vegas    NV
    ## 4 Las Vegas    NV
    ## 5 Las Vegas    NV

**Serendipity**

According to reference 5, The serendipity will be measured as:$Srdp(u) =\\frac{|UNEXP(u) \\cap USEFUL(u))|}{N}$

where USEFUL(u) denotes the useful (relevant) items for user u and N is the size of recommendation set RS(u).An unexpected set of recommendations for user u (UNEXP(u)) is defined as: *U**N**E**X**P*(*u*)=*R**S*(*u*)\\*P**M* where PM is a set of recommendations generated by a primitive model which is assumed of low unexpectedness. RS(u) denotes the top-N recommendations generated by a recommender system for user u. When an element of RS(u) does not belong to PM, it is considered to be unexpected.

``` r
U1827_predict <- data.frame(rbind('Primary' = unlist(pri_rec), 'Primary + Useful' = unlist(c1_rec), 'Primary + Useful + Funny' = unlist(c2_rec), 'Primary + Useful + Funny + Cool' = unlist(c3_rec)))
colnames(U1827_predict) <- paste0("No.",seq(1:10))
kable(U1827_predict)
```

|                                 | No.1  | No.2 | No.3  | No.4  | No.5  | No.6  | No.7  | No.8  | No.9  | No.10 |
|---------------------------------|:------|:-----|:------|:------|:------|:------|:------|:------|:------|:------|
| Primary                         | R1030 | R478 | R745  | R1550 | R1344 | R6798 | R5179 | R228  | R1204 | R229  |
| Primary + Useful                | R1967 | R831 | R603  | R1861 | R1971 | R873  | R5580 | R294  | R1622 | R5977 |
| Primary + Useful + Funny        | R1967 | R603 | R602  | R1081 | R831  | R2291 | R1861 | R3438 | R873  | R1464 |
| Primary + Useful + Funny + Cool | R831  | R603 | R1861 | R1622 | R5580 | R63   | R294  | R6589 | R1464 | R2410 |

``` r
pri_rating <- predict(r_ubcf, mx_r[1201], type="ratings",n=10)
usefulness <- as(pri_rating, "matrix")
usefulness_df <- as.data.frame(usefulness) %>%
    gather(restaurant_id, predicted_rating,1:length(usefulness))

unexpected_1 <- setdiff(pri_rec[[1]], c1_rec[[1]])
unexpected_ratings <- filter(usefulness_df, restaurant_id  %in% unexpected_1 ) %>%
  filter(predicted_rating > mean(mx_r@data@x) )
serendipity_c1 <- nrow(unexpected_ratings)/length(unlist(c1_rec))
print(paste("serendipity for user 1827 using combiantion of primary rating and useful rating is:",serendipity_c1*100,"%"))
```

    ## [1] "serendipity for user 1827 using combiantion of primary rating and useful rating is: 100 %"

By combing primary rating and useful rating, we can get a totally different top 10 recommendations for user 1807.

``` r
unexpected_ratings <- left_join(unexpected_ratings,idf_city,by="restaurant_id")
kable(unexpected_LasVegas <- filter(unexpected_ratings,city == "Las Vegas"))
```

| restaurant\_id |  predicted\_rating|  restaurant\_No| restaurant                    | city      | state |
|:---------------|------------------:|---------------:|:------------------------------|:----------|:------|
| R228           |           4.224769|             228| Bachi Burger                  | Las Vegas | NV    |
| R478           |           4.249639|             478| SkinnyFATS                    | Las Vegas | NV    |
| R1030          |           4.266748|            1030| Desert Wireless iPhone Repair | Las Vegas | NV    |
| R1204          |           4.224138|            1204| The Buffet at Bellagio        | Las Vegas | NV    |
| R5179          |           4.225634|            5179| Lucki Thai                    | Las Vegas | NV    |
| R6798          |           4.232940|            6798| 9037 Salon                    | Las Vegas | NV    |

``` r
new_restaurant <- setdiff(unexpected_LasVegas$restaurant,Lasvegas$restaurant)
print(paste("By combing primary rating and useful rating,we found",length(new_restaurant),"restaurants not recommended by the primary model by relevant:",paste(unlist(new_restaurant), collapse=','),"for user 1827."))
```

    ## [1] "By combing primary rating and useful rating,we found 6 restaurants not recommended by the primary model by relevant: Bachi Burger,SkinnyFATS,Desert Wireless iPhone Repair,The Buffet at Bellagio,Lucki Thai,9037 Salon for user 1827."

``` r
# topN for test data set based on primary recommendation system
(p_topN <- predict(r_ubcf, getData(e,"unknown"),type="topNList",n=10))
```

    ## Recommendations as 'topNList' with n = 10 for 240 users.

``` r
# show predicted top10 restaurants
pri_rec <- as(p_topN, "list")

# topN for test data set based on primary+useful rating
(c1_p_topN <- predict(c1_ubcf, getData(e,"unknown"),type="topNList",n=10))
```

    ## Recommendations as 'topNList' with n = 10 for 240 users.

``` r
# show predicted top10 restaurants
c1_rec <- as(c1_p_topN, "list")

serendipity_c1_df <- data.frame()
for (i in 1:length(pri_rec)){
  unexpected_1 <- setdiff(pri_rec[[i]], c1_rec[[i]])
  unexpected_ratings <- filter(usefulness_df, restaurant_id %in% unexpected_1 ) %>%
    filter(predicted_rating > mean(mx_r@data@x))
  serendipity_c1[i] <- nrow(unexpected_ratings)/10
  serendipity_c1_df_1 <- data.frame('user_id' = names(pri_rec[i]),'serendipity'= serendipity_c1[i])
  serendipity_c1_df <- rbind(serendipity_c1_df,serendipity_c1_df_1)
}

head(serendipity_c1_df, n= 5)
```

    ##   user_id serendipity
    ## 1     U11           1
    ## 2     U31           1
    ## 3     U40           1
    ## 4     U41           1
    ## 5     U43           1

``` r
unexpected_ratings <- filter(usefulness_df, restaurant_id  %in% unexpected_1 ) %>%
  filter(predicted_rating > mean(mx_r@data@x) )
```

### Conclusion and Discussion:

1.  One restaurant recommendation system based on the user\_based collaborative filtering algorithm was built with the Yelp academic data for challenge round 9.The RMSE is 1.47.

2.  Restaurants recommending results could be furthered modified by the location. In the future, that information on locations (such as longitude and latitude) or the distance between restaurants, could be used to calculate the similarity.

3.  The recommendation system based on multi-criteria ratings generated a totally different list of restaurants for users. It is intriguing to see that the serendipity of the recommendation system based on multi-criteria ratings for each user was 100%. At the same time, the accuracy of the prediction was higher than only using one-criteria of rating, The RMSE reduced to 0.5.

### Reference:

1.  Blanca Vargas-Govea, Gabriel González-Serna, Rafael Ponce-Medellín. Effects of relevant contextual features in the performance of a restaurant recommender system.CARS,( 2011)

2.  Mengqi Yu, Meng Xue, Wenjia Ouyang. Restaurants Review Star Prediction for Yelp Dataset.Conference Proceedings (2015).

3.  Gediminas Adomavicius, YoungOk Kwon. New Recommendation Techniques for Multi-Criteria Rating Systems. IEEE Intelligent Systems 22-3 (2017).

4.  Jun Zeng, Feng Li, Haiyang Liu, Junhao Wen, Sachio Hirokawa. A Restaurant Recommender System Based on User Preference and Location in Mobile Environment. Advanced Applied Informatics (IIAI-AAI), 2016 5th IIAI International Congress.

5.  Qiuxia Lu, Tianqi Chen, Weinan Zhang, Diyi Yang, Yong Yu.Serendipitous Personalized Ranking for Top-N Recommendation.Proceeding WI-IAT '12 Proceedings of the 2012 IEEE/WIC/ACM International Joint Conferences on Web Intelligence and Intelligent Agent Technology, Volume 01, 258-265 (2012).
