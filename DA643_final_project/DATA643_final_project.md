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

#### 3.1 Process the raw data

#### 3.2 Explore the data

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


#build the user-item matrix
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

#converting dcGMatrix to realRatingMatrix for applyting recommenderlab
mx <- as(rating_mx,"realRatingMatrix")
#setting itemlabels
colnames(mx) <- paste("R", 1:65432, sep = "")
as(mx[1,1:10], "list")
```

    ## [[1]]
    ##  R1  R2  R3  R4  R5  R6  R7  R8  R9 R10 
    ##   5   5   5   4   5   5   5   5   5   4

``` r
#setting userlabels
rownames(mx) <- paste("U", 1:63081, sep = "")
as(mx[1,1:10], "list")
```

    ## $U1
    ##  R1  R2  R3  R4  R5  R6  R7  R8  R9 R10 
    ##   5   5   5   4   5   5   5   5   5   4

``` r
#Normalize by subtracting the row mean from all ratings in the row
mx_n <- normalize(mx)

#view the matrix
getRatingMatrix(mx)[1:10,1:5]
```

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

``` r
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
```

    ## [1] "Total number of people who had rounded average ratings: 405551"

``` r
user_rate_1 <- sum(user$review_count == 1)
user_rate_2 <- sum(user$review_count == 2)
user_rate_3 <- sum(user$review_count == 3)
user_rate_4 <- sum(user$review_count == 4)
print(paste("Number of people who only rated one restaurant:",user_rate_1))
```

    ## [1] "Number of people who only rated one restaurant: 189809"

``` r
print(paste("Number of people who only rated twice:",user_rate_2))
```

    ## [1] "Number of people who only rated twice: 126347"

``` r
print(paste("Number of people who only rated three times:",user_rate_3))
```

    ## [1] "Number of people who only rated three times: 96815"

``` r
print(paste("Number of people who only rated four times:",user_rate_4))
```

    ## [1] "Number of people who only rated four times: 69627"

``` r
print(paste("Number of people who only rated less than three times:",user_rate_1 + user_rate_2 +user_rate_3))
```

    ## [1] "Number of people who only rated less than three times: 412971"

By viewing the data we see:

1.Rating distribution is not normal with the most frequent rating at the highest rating 5, whose frequency is much higher than other ratings. One possibility is that people who would write reviews for restaurant on Yelp are those who will view review/ratings online before deciding to try a new restaurant. So there is more chance that these people like what they chose. This suggests that the current restaurant recommendation systems work very well so it is more likely that people could find the food they like by searching on the recommender engine/application.

2.Distribution of user review count is not normal with a average at 24. Majority people only wrote a few reviews and there are very few people wrote thousands of reviews with a maximum number at 11284.By looking at the minimum review count we knew that some people did not write any review.

3.Distribution of user review count is not normal with a average at 28. Majority restaurant received a few reviews and there are very few restaurant received thousands of reviews with a maximum number at 6414. If we look at the minimum review count, we can see any restaurant in this data set at least got 3 reviews.

4.The average rating for each user is multimodal distribution. The count ofaverage rating at each round number(stars) are much higher than other not rounded number. In consistant to Figure 1, average rating at 5 has the highest frequency. The possible reasons that a lot of people had a rounded average rating could either be these people only give the same rating for different restaurant and they only rated very few restaurants. It is intresting to notice that the number of people who had rounded average ratings, 405551, is close to the number of people who only rated less than three times, 412971.

5.Similar to the user average rating, the average rating for each restaurant is multimodal distribution. In consistant to Figure 1, average rating at 5 has the highest frequency. One of the possible reason for this pattern is there were a large number of restaurant received very few ratings and ratings were the same. Another reason is that there are a lot of very good restaurants always received 5. But, is it really possible?

### 4.Creating a recommender

#### 4.1 Building a User-based Collaborative Filtering Model

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
# r_ibcf <- Recommender(getData(e, "train"), "IBCF",parameter = list(k=30, method = "cosine", normalize = "Z-score", alpha=0.5))

# release memory
gc()
```

    ##            used  (Mb) gc trigger  (Mb)  max used  (Mb)
    ## Ncells  3711240 198.3   12002346 641.0  12002346 641.0
    ## Vcells 55749305 425.4  112780893 860.5 112780892 860.5

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
    ##   1  [0.1sec/11.58sec] 
    ##   2  [0.03sec/12.03sec] 
    ##   3  [0.03sec/12.1sec] 
    ##   4  [0.05sec/11.74sec] 
    ##   5  [0.04sec/11.98sec]

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
as(p_rating, "matrix")[1:10,1:10]
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
    ##          R12      R13      R17
    ## U11 4.429356 4.445291 4.371860
    ## U31 2.732841 2.811651 2.800000
    ## U40 3.754724 4.148015 4.057790
    ## U41 4.524867 4.443738 4.316639
    ## U43 3.767197 3.835270 3.885991
    ## U45 4.244129 4.200000 4.241087
    ## U46 2.523405 2.412146 2.291994
    ## U53 3.274084 3.405942 3.270816
    ## U54 3.620108 3.600000 3.590404
    ## U59 3.803884 3.788111 3.902065

``` r
# RMSE
(error <- data.frame(calcPredictionAccuracy(p_rating, getData(e, "unknown"))))
```

    ##      calcPredictionAccuracy.p_rating..getData.e...unknown...
    ## RMSE                                                1.468892
    ## MSE                                                 2.157643
    ## MAE                                                 1.188701

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
(p_topN <- predict(r_ubcf, mx_r[1201],type="topNList",n=10))
```

    ## Recommendations as 'topNList' with n = 10 for 1 users.

``` r
# show predicted top10 restaurants
as(p_topN, "list")
```

    ## $U1827
    ##  [1] "R1030" "R478"  "R745"  "R1550" "R1344" "R6798" "R5179" "R228" 
    ##  [9] "R1204" "R229"

On practical scenario, we have to consider the location while designing a restaurant recommendation system. In most of the time people will use recommendation engine to find restaurant from a certain city.

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

#get 100 restaurants for User 1201 from recemmender system
(p_top100 <- predict(r_ubcf, mx_r[1201],type="topNList",n=100))
```

    ## Recommendations as 'topNList' with n = 100 for 1 users.

``` r
# filter the restaurant for User 1201 based on location
pred_restaurant <-  data.frame(as(p_top100, "list"))
colnames(pred_restaurant) <- "U1201"
pred_restaurant[] <- lapply(pred_restaurant, as.character)
pred_restaurant$restaurant_id <- pred_restaurant$U1201

pred_restaurant <- left_join(pred_restaurant,idf_city, by='restaurant_id' )
pred_restaurant$city <- as.character(pred_restaurant$city)
pred_restaurant$state <- as.character(pred_restaurant$state)

# For example, if user 1201 want to find restaurant in Las vegas,
(Lasvegas <- filter(pred_restaurant,city == "Las Vegas"))
```

    ##     U1201 restaurant_id restaurant_No                         restaurant
    ## 1   R1030         R1030          1030      Desert Wireless iPhone Repair
    ## 2    R478          R478           478                         SkinnyFATS
    ## 3   R6798         R6798          6798                         9037 Salon
    ## 4   R5179         R5179          5179                         Lucki Thai
    ## 5    R228          R228           228                       Bachi Burger
    ## 6   R1204         R1204          1204             The Buffet at Bellagio
    ## 7   R1483         R1483          1483                          The Henry
    ## 8    R246          R246           246                           Sake Rok
    ## 9    R844          R844           844           Jean Philippe Patisserie
    ## 10   R808          R808           808           Gangnam Asian BBQ Dining
    ## 11  R5370         R5370          5370              Libre Mexican Cantina
    ## 12  R4161         R4161          4161         El Sombrero Mexican Bistro
    ## 13  R1549         R1549          1549                               Cleo
    ## 14    R43           R43            43                      Vintner Grill
    ## 15   R811          R811           811        Cirque du Soleil - Zumanity
    ## 16  R2314         R2314          2314 Rise & Shine - A Steak & Egg Place
    ## 17  R1147         R1147          1147                  Soho SushiBurrito
    ## 18  R3879         R3879          3879              Professor Nails & Spa
    ## 19  R2639         R2639          2639                        Today Nails
    ## 20  R3688         R3688          3688                             Yassou
    ## 21 R10945        R10945         10945        Sun Buggy & ATV Fun Rentals
    ## 22   R123          R123           123                            Topgolf
    ## 23  R1866         R1866          1866             Gallagher's Steakhouse
    ## 24  R1974         R1974          1974             Amore Taste of Chicago
    ## 25  R1998         R1998          1998               WAX Hair Removal Bar
    ## 26  R4425         R4425          4425              La Bonita Supermarket
    ## 27  R6486         R6486          6486                     Cafe Americano
    ## 28   R559          R559           559                     Freed's Bakery
    ## 29    R21           R21            21      Mount Everest India's Cuisine
    ## 30   R491          R491           491                            Egg & I
    ## 31   R563          R563           563                           Oh Curry
    ## 32  R1989         R1989          1989          Don Tortaco Mexican Grill
    ## 33  R7308         R7308          7308                     No. 1 Boba Tea
    ## 34  R2728         R2728          2728                Archi's Thai Bistro
    ## 35 R24446        R24446         24446                         Algobertos
    ## 36  R1449         R1449          1449                       Viet Kitchen
    ##         city state
    ## 1  Las Vegas    NV
    ## 2  Las Vegas    NV
    ## 3  Las Vegas    NV
    ## 4  Las Vegas    NV
    ## 5  Las Vegas    NV
    ## 6  Las Vegas    NV
    ## 7  Las Vegas    NV
    ## 8  Las Vegas    NV
    ## 9  Las Vegas    NV
    ## 10 Las Vegas    NV
    ## 11 Las Vegas    NV
    ## 12 Las Vegas    NV
    ## 13 Las Vegas    NV
    ## 14 Las Vegas    NV
    ## 15 Las Vegas    NV
    ## 16 Las Vegas    NV
    ## 17 Las Vegas    NV
    ## 18 Las Vegas    NV
    ## 19 Las Vegas    NV
    ## 20 Las Vegas    NV
    ## 21 Las Vegas    NV
    ## 22 Las Vegas    NV
    ## 23 Las Vegas    NV
    ## 24 Las Vegas    NV
    ## 25 Las Vegas    NV
    ## 26 Las Vegas    NV
    ## 27 Las Vegas    NV
    ## 28 Las Vegas    NV
    ## 29 Las Vegas    NV
    ## 30 Las Vegas    NV
    ## 31 Las Vegas    NV
    ## 32 Las Vegas    NV
    ## 33 Las Vegas    NV
    ## 34 Las Vegas    NV
    ## 35 Las Vegas    NV
    ## 36 Las Vegas    NV

#### 4.2 Multi-Criteria Recommender System

In the Yelp dataset there is more information other than only ratings, so we can not only use content-based algorithm but also collaborative filtering algorithms. Location of the restaurant is an important factor to do the recommendation, so the location will be considered so the similarity between the distance and similarity between user/items will be combined. Other algorithms like alternative linear squares and singular value decomposition will also be used to build the prediction models.

Because there are three criteria in reviews: funny, useful, and cool, the rating will be calculated as follows:

*R* : *U**s**e**r**s* × *I**t**e**m**s* → *R*<sub>0</sub> × *R*<sub>1</sub> × ...*R*<sub>*k*</sub>

*R*<sub>0</sub> is the set of possible overall rating values, and *R*<sub>*i*</sub> represents the possible rating values for each individual criterion i (i = 1,..,k), typically on some numeric scale.

The prediction results of single-criteria collaborative filtering algorithm and multi-criteria collaborative filtering algorithms will be compared to decide which approach is better.

The implementation and evaluation will be performed in R and Apache Spark. At last, if time permits, an application will be built with the Shiny package.

#### 4.2.1 Building the User-item Matrix Based on Useful, Funny, and Cool Comments

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

From the summary, we can see the value of useful, funny or cool represent how many people felt the reviews for the restaurant are useful, funny or cool. We will take these number as a rating for whether a restaurant is popular. Because the scale is different for these three factors, we will converted them to binary value based on the average.

``` r
# the frequeny of restaurant's rating is useful 
useful_tb <- as.data.frame(table(rating$useful))
useful_tb$Var1 <- as.numeric(as.character(useful_tb$Var1))
# how many pepople believed that the review was useful at a threshold at 100 restaurants having the same amount of "useful" notes
u_threshold <- useful_tb[useful_tb$Freq > 100,]

# the frequeny of restaurant's rating is useful 
funny_tb <- as.data.frame(table(rating$funny))
funny_tb$Var1 <- as.numeric(as.character(funny_tb$Var1))
# how many pepople believed that the review was useful at a threshold at 100 restaurants having the same amount of "useful" notes
f_threshold <- funny_tb[funny_tb$Freq > 100,]

# the frequeny of restaurant's rating is useful 
cool_tb <- as.data.frame(table(rating$useful))
cool_tb$Var1 <- as.numeric(as.character(cool_tb$Var1))
# how many pepople believed that the review was useful at a threshold at 100 restaurants having the same amount of "useful" notes
c_threshold <- cool_tb[cool_tb$Freq > 100,]

# convert the basic rating matrix to binary matrix
mx_r@data@x [mx_r@data@x > mean(mx_r@data@x[])]<- 1
mx_r@data@x [mx_r@data@x < mean(mx_r@data@x[])]<- 0

# convert the useful matrix to binary matrix
u_mx@data@x [u_mx@data@x > max(u_threshold$Var1)]<- 1
u_mx@data@x [u_mx@data@x < max(u_threshold$Var1)]<- 0

# convert the funny rating matrix to binary matrix  
f_mx@data@x [f_mx@data@x > max(f_threshold$Var1)]<- 1
f_mx@data@x [f_mx@data@x < max(f_threshold$Var1)]<- 0

# convert the cool rating matrix to binary matrix
c_mx@data@x [c_mx@data@x > max(c_threshold$Var1)]<- 1
c_mx@data@x [c_mx@data@x < max(c_threshold$Var1)]<- 0


#chose the users and restaurants matching the constrained user-item matrix which users rated the restaurant more than 20 times and restaurants received more than 50 reviews.
u_mx_fit <- u_mx[,c(colnames(mx_r))]
u_mx_fit <- u_mx_fit[row.names(u_mx_fit) %in% c(rownames(mx_r)),]

f_mx_fit <- f_mx[,c(colnames(mx_r))]
f_mx_fit <- f_mx_fit[row.names(f_mx_fit) %in% c(rownames(mx_r)),]

c_mx_fit <- c_mx[,c(colnames(mx_r))]
c_mx_fit <- c_mx_fit[row.names(c_mx_fit) %in% c(rownames(mx_r)),]

# Element-wise multiplication
r0_r1 <- mx_r@data * u_mx_fit@data
r0_r1_r2 <- r0_r1 * f_mx_fit@data
r0_r1_r2_r3 <- r0_r1_r2 * c_mx_fit@data
```

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
    ## Ncells  3731879 199.4   12002346 641.0  12002346 641.0
    ## Vcells 77491737 591.3  112780893 860.5 112780892 860.5

``` r
# evaluation  
c1_results <- evaluate(c1_e, method="UBCF", type = "ratings", n=c(1,3,5,10,15,20))
```

    ## UBCF run fold/sample [model time/prediction time]
    ##   1  [0.05sec/13.63sec] 
    ##   2  [0.03sec/13.84sec] 
    ##   3  [0.03sec/13.05sec] 
    ##   4  [0.2sec/13.45sec] 
    ##   5  [0.04sec/13.05sec]

``` r
avg(results)
```

    ##         RMSE      MSE      MAE
    ## res 1.485573 2.207031 1.221872

``` r
# making predictions on ratings
(c1_p_rating <- predict(c1_ubcf, getData(c1_e, "known"), type="ratings",n=10))
```

    ## 240 x 5243 rating matrix of class 'realRatingMatrix' with 1257120 ratings.

``` r
# show predicted ratings
as(c1_p_rating, "matrix")[1:10,1:10]
```

    ##     R1 R3 R4 R5 R6 R10 R11 R12 R13 R17
    ## U6   0  0  0  0  0   0   0   0   0   0
    ## U14  0  0  0  0  0   0   0   0   0   0
    ## U30  0  0  0  0  0   0   0   0   0   0
    ## U33  0  0  0  0  0   0   0   0   0   0
    ## U40  0  0  0  0  0   0   0   0   0   0
    ## U41  0  0  0  0  0   0   0   0   0   0
    ## U44  0  0  0  0  0   0  NA   0   0   0
    ## U57  0  0  0  0  0   0   0   0   0   0
    ## U58  0  0  0  0  0   0   0   0   0   0
    ## U66  0  0  0  0  0   0   0   0   0   0

``` r
# RMSE
(error <- data.frame(calcPredictionAccuracy(c1_p_rating, getData(c1_e, "unknown"))))
```

    ##      calcPredictionAccuracy.c1_p_rating..getData.c1_e...unknown...
    ## RMSE                                                   0.343959123
    ## MSE                                                    0.118307878
    ## MAE                                                    0.001460591

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
as(c1_p_topN, "list")
```

    ## $U1827
    ##  [1] "R1"  "R3"  "R4"  "R5"  "R6"  "R10" "R11" "R12" "R13" "R17"

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
    ## Ncells  3732259 199.4   12002346  641.0  12002346  641.0
    ## Vcells 81477450 621.7  162580485 1240.4 162578332 1240.4

``` r
# evaluation  
c2_results <- evaluate(c2_e, method="UBCF", type = "ratings", n=c(1,3,5,10,15,20))
```

    ## UBCF run fold/sample [model time/prediction time]
    ##   1  [0.07sec/12.37sec] 
    ##   2  [0.05sec/13sec] 
    ##   3  [0.05sec/12.94sec] 
    ##   4  [0.06sec/12.78sec] 
    ##   5  [0.05sec/12.82sec]

``` r
avg(results)
```

    ##         RMSE      MSE      MAE
    ## res 1.485573 2.207031 1.221872

``` r
# making predictions on ratings
(c2_p_rating <- predict(c2_ubcf, getData(c2_e, "known"), type="ratings",n=10))
```

    ## 240 x 5243 rating matrix of class 'realRatingMatrix' with 1257120 ratings.

``` r
# show predicted ratings
as(c2_p_rating, "matrix")[1:10,1:10]
```

    ##     R1 R3 R4 R5 R6 R10 R11 R12 R13 R17
    ## U8   0  0  0  0  0   0   0   0   0   0
    ## U24  0  0  0  0  0   0   0   0   0   0
    ## U26  0  0  0  0  0   0   0   0   0   0
    ## U29  0  0  0  0  0   0   0   0   0   0
    ## U36  0  0  0  0  0   0   0   0   0   0
    ## U37  0  0  0  0  0   0   0   0   0   0
    ## U55  0  0  0  0  0   0   0   0   0   0
    ## U65  0  0  0  0  0   0   0   0   0   0
    ## U71  0  0  0  0  0   0   0   0   0   0
    ## U75  0  0  0  0  0   0   0   0   0   0

``` r
# RMSE
(error <- data.frame(calcPredictionAccuracy(c2_p_rating, getData(c2_e, "unknown"))))
```

    ##      calcPredictionAccuracy.c2_p_rating..getData.c2_e...unknown...
    ## RMSE                                                             0
    ## MSE                                                              0
    ## MAE                                                              0

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
as(c2_p_topN, "list")
```

    ## $U1827
    ##  [1] "R1"  "R3"  "R4"  "R5"  "R6"  "R10" "R11" "R12" "R13" "R17"

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
    ## Ncells  3732540 199.4   12002346  641.0  12002346  641.0
    ## Vcells 85453623 652.0  162580485 1240.4 162579271 1240.4

``` r
# evaluation  
c3_results <- evaluate(c3_e, method="UBCF", type = "ratings", n=c(1,3,5,10,15,20))
```

    ## UBCF run fold/sample [model time/prediction time]
    ##   1  [0.05sec/12.64sec] 
    ##   2  [0.03sec/13.74sec] 
    ##   3  [0.03sec/13.14sec] 
    ##   4  [0.04sec/13.06sec] 
    ##   5  [0.05sec/13.79sec]

``` r
avg(results)
```

    ##         RMSE      MSE      MAE
    ## res 1.485573 2.207031 1.221872

``` r
# making predictions on ratings
(c3_p_rating <- predict(c3_ubcf, getData(c3_e, "known"), type="ratings",n=10))
```

    ## 240 x 5243 rating matrix of class 'realRatingMatrix' with 1257120 ratings.

``` r
# show predicted ratings
as(c3_p_rating, "matrix")[1:10,1:10]
```

    ##     R1 R3 R4 R5 R6 R10 R11 R12 R13 R17
    ## U5   0  0  0  0  0   0   0   0   0   0
    ## U15  0  0  0  0  0   0   0   0   0   0
    ## U24  0  0  0  0  0   0   0   0   0   0
    ## U26  0  0  0  0  0   0   0   0   0   0
    ## U34  0  0  0  0  0   0   0   0   0   0
    ## U36  0  0  0  0  0   0   0   0   0   0
    ## U46  0  0  0  0  0   0   0   0   0   0
    ## U58  0  0  0  0  0   0   0   0   0   0
    ## U59  0  0  0  0  0   0   0   0   0   0
    ## U71  0  0  0  0  0   0   0   0   0   0

``` r
# RMSE
(error <- data.frame(calcPredictionAccuracy(c3_p_rating, getData(c3_e, "unknown"))))
```

    ##      calcPredictionAccuracy.c3_p_rating..getData.c3_e...unknown...
    ## RMSE                                                             0
    ## MSE                                                              0
    ## MAE                                                              0

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
as(c3_p_topN, "list")
```

    ## $U1827
    ##  [1] "R1"  "R3"  "R4"  "R5"  "R6"  "R10" "R11" "R12" "R13" "R17"

### Reference:

1.  Blanca Vargas-Govea, Gabriel González-Serna, Rafael Ponce-Medellín. Effects of relevant contextual features in the performance of a restaurant recommender system.CARS,( 2011)
2.  Mengqi Yu, Meng Xue, Wenjia Ouyang. Restaurants Review Star Prediction for Yelp Dataset.Conference Proceedings (2015).
3.  Gediminas Adomavicius, YoungOk Kwon. New Recommendation Techniques for Multi-Criteria Rating Systems. IEEE Intelligent Systems 22-3 (2017).
4.  Jun Zeng, Feng Li, Haiyang Liu, Junhao Wen, Sachio Hirokawa. A Restaurant Recommender System Based on User Preference and Location in Mobile Environment. Advanced Applied Informatics (IIAI-AAI), 2016 5th IIAI International Congress.
