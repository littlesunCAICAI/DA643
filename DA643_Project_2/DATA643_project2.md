DATA 643 Project 2 - Content-Based and Collaborative Filtering
================
Yun Mai
June 17, 2017

In this project I will try out different ways of implementing and configuring a recommender, and to evaluate different approaches.

I will use a movie ratings dataset built for an assignment of ralational database to implement the following recommendation algorithms:

. Content-Based Filtering

. User-User Collaborative Filtering

. Item-Item Collaborative Filtering

To build a Content-Based recommender, I will use genre as features for modeling. Because there are only 7 movies, I copy-pasted the storyline from IMDB in case tf-idf will be performed to do topic modeling.

To build a callaborative filtering recommender, recommenderlab package will be used.

I will evaluate and compare the perfomance of item-based and user-based callaborative filtering approaches.

Overview:

Data

1.  Content-Based Filtering

    1.1 Binary Representation

    1.1.1 Binary Feature Matrix

        1.1.1.1 Feature Matrix

        1.1.1.2 Document Frequency (DF) and Inverse Document Frequency (IDF)

        1.1.1.3 Total_atrributes

        1.1.1.4 bianary rating matrix 

    1.1.2 Normoalization of Fetures Matrix

        1.1.3 User Profile

        1.1.4 Weighted Scores

        1.1.5 Prediction

    1.2 Non-binary representation

        1.2.1 Feature extraction

        1.2.2 Pridiction

2.  Collaborative Filtering

    2.1 Coercion the Data to Rating Matrices

    2.2 Normalization

    2.3 IBCF: Item-Based Collaborative Filtering

    2.4 UBCF: User-Based Collaborative Filtering

    2.5 Evaluation of Predicted Ratings

    2.6 Evaluation of a top-N Recommender Algorithm

Set up working environment.

``` r
install.packages("R.matlab")
install.packages("recommenderlab")
install.packages("tidytext")
install.packages("janeaustenr")
```

Load packages.

``` r
suppressWarnings(suppressMessages(library(RCurl)))
suppressWarnings(suppressMessages(library(knitr)))
suppressWarnings(suppressMessages(library(tidyr)))
suppressWarnings(suppressMessages(library(stringr)))
suppressWarnings(suppressMessages(library(R.matlab)))
# the following three packages will be used for extract the features from movie stroyline
# creat tbl_df, tbl from dataframe for tidytext
suppressWarnings(suppressMessages(library(tibble)))
# split words from movie text
suppressWarnings(suppressMessages(library(tidytext)))
# count words
suppressWarnings(suppressMessages(library(dplyr)))

# recommenderlab will be used for collaborative filtering 
suppressWarnings(suppressMessages(library(recommenderlab)))
# draw figures
suppressWarnings(suppressMessages(library(ggplot2)))
```

#### Data

``` r
# load data of movie ratings from my friends created for investigating rational database 
url <- "https://raw.githubusercontent.com/YunMai-SPS/DA643/master/DA643_Project_2/OMDB_data/omdb_2.csv"
url_rating <- "https://raw.githubusercontent.com/YunMai-SPS/DA643/master/DA643_Project_2/OMDB_data/rating_2.csv"  
url_friend <- "https://raw.githubusercontent.com/YunMai-SPS/DA643/master/DA643_Project_2/OMDB_data/friends.csv"
url_genre <- "https://raw.githubusercontent.com/YunMai-SPS/DA643/master/DA643_Project_2/OMDB_data/genre.csv"
kable(head(movie <- read.csv(url),n=2))
```

|  row\_names| Title                    |  Year| imdbID    | genre                      | actor                                                    | country                    | director     | writer                                                            |  MovieID|  box\_office| storyline                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
|-----------:|:-------------------------|-----:|:----------|:---------------------------|:---------------------------------------------------------|:---------------------------|:-------------|:------------------------------------------------------------------|--------:|------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|           1| The Fate of the Furious  |  2017| tt4630562 | Action,Crime,Thriller      | Scott Eastwood,Charlize Theron,Dwayne Johnson,Vin Diesel | USA,France,Canada,UK,Samoa | F. Gary Gray | Chris Morgan,Gary Scott Thompson (based on characters created by) |        1|    224507635| Now that Dom and Letty are on their honeymoon and Brian and Mia have retired from the game-and the rest of the crew has been exonerated-the globetrotting team has found a semblance of a normal life. But when a mysterious woman seduces Dom into the world of crime he can't seem to escape and a betrayal of those closest to him, they will face trials that will test them as never before. From the shores of Cuba and the streets of New York City to the icy plains off the arctic Barents Sea, the elite force will crisscross the globe to stop an anarchist from unleashing chaos on the world's stage... and to bring home the man who made them a family. |
|           2| Star Wars: The Last Jedi |  2017| tt2527336 | Action, Adventure, Fantasy | Tom Hardy, Daisy Ridley, Adam Driver, Mark Hamill        | USA                        | Rian Johnson | Rian Johnson (screenplay), George Lucas (characters)              |        2|           NA| Having taken her first steps into a larger world in Star Wars: The Force Awakens (2015), Rey continues her epic journey with Finn, Poe and Luke Skywalker in the next chapter of the saga.                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |

``` r
kable(head(rating <- read.csv(url_rating),n=2))
```

|  row\_names|  MovieID| MovieName               |  FriendID| FriendName |  FriendRating|
|-----------:|--------:|:------------------------|---------:|:-----------|-------------:|
|           1|        1| The Fate of the Furious |         1| Ming       |             2|
|           8|        1| The Fate of the Furious |         2| Hao        |            NA|

``` r
kable(head(friend <- read.csv(url_friend),n=2))
```

|  row\_names|  FriendsID| FriendsName |
|-----------:|----------:|:------------|
|           1|          1| Ming        |
|           2|          2| Hao         |

``` r
kable(head(genre <- read.csv(url_genre),n=2))
```

|  row\_names| genres | title                   |  genreID|
|-----------:|:-------|:------------------------|--------:|
|           1| Action | The Fate of the Furious |        1|
|           2| Crime  | The Fate of the Furious |        2|

``` r
rating <- merge(rating,friend, by.x = "FriendID", by.y = "FriendsID")
kable(head(rating <- rating[c("FriendID", "FriendName", "MovieID", "MovieName", "FriendRating")],n=2))
```

|  FriendID| FriendName |  MovieID| MovieName               |  FriendRating|
|---------:|:-----------|--------:|:------------------------|-------------:|
|         1| Ming       |        1| The Fate of the Furious |             2|
|         1| Ming       |        6| Logan                   |             5|

#### **1. Content-Based Filtering**

#### 1.1 Binary Representation

#### 1.1.1.1 Feature Matrix

Movie genre will be used as the fesatures to build the content-based filtering. To create a feature matrix, the genre table will be reshaped.

``` r
genre_2 <- genre[c("genres","genreID")]
genre_2 <- genre_2[!duplicated(genre_2$genres),]
genre_2$genreID <- seq(1:nrow(genre_2))
genre_3 <- subset(genre, select = -c(row_names,genreID))
genre_4 <- merge(genre_2,genre_3, by.x = "genres", by.y = "genres")
genre_5 <- spread(genre_4, genreID, genres)
colnames(genre_5) <- c('title',as.character(genre_2$genres))
genre_5[is.na(genre_5)] <- 0
```

    ## Warning in `[<-.factor`(`*tmp*`, thisvar, value = 0): invalid factor level,
    ## NA generated

    ## Warning in `[<-.factor`(`*tmp*`, thisvar, value = 0): invalid factor level,
    ## NA generated

    ## Warning in `[<-.factor`(`*tmp*`, thisvar, value = 0): invalid factor level,
    ## NA generated

    ## Warning in `[<-.factor`(`*tmp*`, thisvar, value = 0): invalid factor level,
    ## NA generated

    ## Warning in `[<-.factor`(`*tmp*`, thisvar, value = 0): invalid factor level,
    ## NA generated

    ## Warning in `[<-.factor`(`*tmp*`, thisvar, value = 0): invalid factor level,
    ## NA generated

    ## Warning in `[<-.factor`(`*tmp*`, thisvar, value = 0): invalid factor level,
    ## NA generated

    ## Warning in `[<-.factor`(`*tmp*`, thisvar, value = 0): invalid factor level,
    ## NA generated

    ## Warning in `[<-.factor`(`*tmp*`, thisvar, value = 0): invalid factor level,
    ## NA generated

    ## Warning in `[<-.factor`(`*tmp*`, thisvar, value = 0): invalid factor level,
    ## NA generated

    ## Warning in `[<-.factor`(`*tmp*`, thisvar, value = 0): invalid factor level,
    ## NA generated

``` r
a <- genre_5[,2:12]
a$Action <- str_replace_all(a$Action,"[[:alpha:]].+","1")
a$Crime <- str_replace_all(a$Crime,"[[:alpha:]].+","1")
a$Thriller <- str_replace_all(a$Thriller,"[[:alpha:]].+","1")
a$Adventure <- str_replace_all(a$Adventure,"[[:alpha:]].+","1")
a$Fantasy <- str_replace_all(a$Fantasy,"[[:alpha:]].+","1")
a[,'Sci-Fi'] <- str_replace_all(a[,'Sci-Fi'],"[[:alpha:]].+","1")
a$Family <- str_replace_all(a$Family,"[[:alpha:]].+","1")
a$Musical <- str_replace_all(a$Musical,"[[:alpha:]].+","1")
a$Drama <- str_replace_all(a$Drama,"[[:alpha:]].+","1")
a$Animation <- str_replace_all(a$Animation,"[[:alpha:]].+","1")
a$Comedy <- str_replace_all(a$Comedy,"[[:alpha:]].+","1")
a[is.na(a)] <- 0
binary_genre_matrix <- sapply(a,as.numeric)
kable(binary_genre_matrix_df <- cbind('MovieName'= genre_5[,1],a))
```

| MovieName                    | Action | Crime | Thriller | Adventure | Fantasy | Sci-Fi | Family | Musical | Drama | Animation | Comedy |
|:-----------------------------|:-------|:------|:---------|:----------|:--------|:-------|:-------|:--------|:------|:----------|:-------|
| Beauty and the Beast         | 0      | 0     | 0        | 0         | 1       | 0      | 1      | 1       | 0     | 0         | 0      |
| Guardians of the Galaxy Vol. | 1      | 0     | 0        | 1         | 0       | 1      | 0      | 0       | 0     | 0         | 0      |
| Logan                        | 1      | 0     | 0        | 0         | 0       | 1      | 0      | 0       | 1     | 0         | 0      |
| Star Wars: The Last Jedi     | 1      | 0     | 0        | 1         | 1       | 0      | 0      | 0       | 0     | 0         | 0      |
| The Fate of the Furious      | 1      | 1     | 1        | 0         | 0       | 0      | 0      | 0       | 0     | 0         | 0      |
| The Good Dinosaur            | 0      | 0     | 0        | 1         | 1       | 0      | 1      | 0       | 1     | 1         | 1      |
| Thor: Ragnarok               | 1      | 0     | 0        | 1         | 1       | 0      | 0      | 0       | 0     | 0         | 0      |

#### 1.1.1.2 Document Frequency (DF) and Inverse Document Frequency (IDF)

``` r
binary_DF <- colSums(binary_genre_matrix,na.rm = T, dims = 1)
N <- 7
binary_IDF <- log10(N/binary_DF)
kable(binary_DF_IDF <- data.frame(binary_DF,binary_IDF))
```

|           |  binary\_DF|  binary\_IDF|
|-----------|-----------:|------------:|
| Action    |           5|     0.146128|
| Crime     |           1|     0.845098|
| Thriller  |           1|     0.845098|
| Adventure |           4|     0.243038|
| Fantasy   |           4|     0.243038|
| Sci-Fi    |           2|     0.544068|
| Family    |           2|     0.544068|
| Musical   |           1|     0.845098|
| Drama     |           2|     0.544068|
| Animation |           1|     0.845098|
| Comedy    |           1|     0.845098|

#### 1.1.1.3 Total\_atrributes

``` r
binary_Total_atrributes <- rowSums(binary_genre_matrix,na.rm = T, dims = 1)
kable(binary_Total_atrributes_df <- data.frame('Moive'= binary_genre_matrix_df[,'MovieName'], binary_Total_atrributes))
```

| Moive                        |  binary\_Total\_atrributes|
|:-----------------------------|--------------------------:|
| Beauty and the Beast         |                          3|
| Guardians of the Galaxy Vol. |                          3|
| Logan                        |                          3|
| Star Wars: The Last Jedi     |                          3|
| The Fate of the Furious      |                          3|
| The Good Dinosaur            |                          6|
| Thor: Ragnarok               |                          3|

#### 1.1.1.4 bianary rating matrix

``` r
rating_1 <- rating[c('FriendName', 'MovieName', 'FriendRating')]
binary_rating_matrix <- spread(rating_1, MovieName, FriendRating)
kable(head(binary_rating_matrix,n=3))
```

| FriendName |  Beauty and the Beast| Guardians of the Galaxy Vol. |  Logan|  Star Wars: The Last Jedi|  The Fate of the Furious|  The Good Dinosaur|  Thor: Ragnarok|
|:-----------|---------------------:|:----------------------------:|------:|-------------------------:|------------------------:|------------------:|---------------:|
| Alison     |                     5|               2              |    1.0|                       1.5|                      4.5|                 NA|              NA|
| Eran       |                     5|              NA              |    1.5|                       1.0|                      3.0|                 NA|               2|
| Hao        |                     4|               3              |    1.5|                       4.5|                       NA|                  5|               4|

``` r
binary_rating_mean <- matrix(rowMeans(binary_rating_matrix[,2:8], na.rm = T, dims = 1))
binary_rating_mean_df <- as.data.frame(binary_rating_mean)
binary_rating_mean_df$Friend <- binary_rating_matrix[,'FriendName']
colnames(binary_rating_mean_df) <- c('rating_mean','Friend')
kable(binary_rating_mean_df <- binary_rating_mean_df[,c('Friend','rating_mean')])
```

| Friend |  rating\_mean|
|:-------|-------------:|
| Alison |      2.800000|
| Eran   |      2.500000|
| Hao    |      3.666667|
| Kate   |      5.000000|
| Mike   |      3.000000|
| Ming   |      3.300000|
| Orshi  |      2.000000|
| Tito   |      4.125000|

``` r
binary_rating_mean_mx <- matrix(rep(binary_rating_mean, 7),nrow=8,ncol=7)
binary_rating_matrix_intm <- binary_rating_matrix[,2:8] - binary_rating_mean_mx
binary_rating_matrix_nor <- sapply(binary_rating_matrix_intm, function(x) ifelse(x > 0, 1, -1))
binary_rating_matrix_nor_df <- as.data.frame(binary_rating_matrix_nor)
binary_rating_matrix_nor_df$Friend <- as.character(binary_rating_mean_df[,1])
kable(binary_rating_matrix_nor_df <- binary_rating_matrix_nor_df[,c(8,1:7)])
```

| Friend |  Beauty and the Beast| Guardians of the Galaxy Vol. |  Logan|  Star Wars: The Last Jedi|  The Fate of the Furious|  The Good Dinosaur|  Thor: Ragnarok|
|:-------|---------------------:|:----------------------------:|------:|-------------------------:|------------------------:|------------------:|---------------:|
| Alison |                     1|              -1              |     -1|                        -1|                        1|                 NA|              NA|
| Eran   |                     1|              NA              |     -1|                        -1|                        1|                 NA|              -1|
| Hao    |                     1|              -1              |     -1|                         1|                       NA|                  1|               1|
| Kate   |                    -1|              NA              |     NA|                        NA|                       NA|                 -1|              NA|
| Mike   |                    -1|              -1              |      1|                         1|                       -1|                 -1|              NA|
| Ming   |                     1|               1              |      1|                        NA|                       -1|                 NA|              -1|
| Orshi  |                    NA|              -1              |     NA|                        -1|                        1|                 NA|              -1|
| Tito   |                    NA|               1              |     -1|                        -1|                       NA|                 NA|              -1|

#### 1.1.2 Normoalization of Fetures Matrix

For a binary data, we nomalize the item profile by deviding the term occurrence(1/0) by the square root of number of features in the movie.

``` r
x <- matrix(rep(sqrt(binary_Total_atrributes),11),7)
binary_genre_matrix_nor <- binary_genre_matrix / x
binary_genre_matrix_nor_df <- data.frame(binary_genre_matrix_nor)
binary_genre_matrix_nor_df$Movie <- binary_genre_matrix_df[,1]
(binary_genre_matrix_nor_df <- binary_genre_matrix_nor_df[,c(12,1:11)])
```

    ##                           Movie    Action     Crime  Thriller Adventure
    ## 1          Beauty and the Beast 0.0000000 0.0000000 0.0000000 0.0000000
    ## 2 Guardians of the Galaxy Vol.  0.5773503 0.0000000 0.0000000 0.5773503
    ## 3                         Logan 0.5773503 0.0000000 0.0000000 0.0000000
    ## 4      Star Wars: The Last Jedi 0.5773503 0.0000000 0.0000000 0.5773503
    ## 5       The Fate of the Furious 0.5773503 0.5773503 0.5773503 0.0000000
    ## 6             The Good Dinosaur 0.0000000 0.0000000 0.0000000 0.4082483
    ## 7                Thor: Ragnarok 0.5773503 0.0000000 0.0000000 0.5773503
    ##     Fantasy    Sci.Fi    Family   Musical     Drama Animation    Comedy
    ## 1 0.5773503 0.0000000 0.5773503 0.5773503 0.0000000 0.0000000 0.0000000
    ## 2 0.0000000 0.5773503 0.0000000 0.0000000 0.0000000 0.0000000 0.0000000
    ## 3 0.0000000 0.5773503 0.0000000 0.0000000 0.5773503 0.0000000 0.0000000
    ## 4 0.5773503 0.0000000 0.0000000 0.0000000 0.0000000 0.0000000 0.0000000
    ## 5 0.0000000 0.0000000 0.0000000 0.0000000 0.0000000 0.0000000 0.0000000
    ## 6 0.4082483 0.0000000 0.4082483 0.0000000 0.4082483 0.4082483 0.4082483
    ## 7 0.5773503 0.0000000 0.0000000 0.0000000 0.0000000 0.0000000 0.0000000

#### 1.1.3 User Profile

``` r
binary_rating_matrix_nor[is.na(binary_rating_matrix_nor)] <- 0
binary_user_profile <- binary_rating_matrix_nor %*% binary_genre_matrix_nor
binary_user_profile_df <- as.data.frame(binary_user_profile)
binary_user_profile_df$Friend <- binary_rating_matrix_nor_df[,1]
(binary_user_profile_df <- binary_user_profile_df[,c(12,1:11)])
```

    ##   Friend    Action      Crime   Thriller  Adventure    Fantasy     Sci-Fi
    ## 1 Alison -1.154701  0.5773503  0.5773503 -1.1547005  0.0000000 -1.1547005
    ## 2   Eran -1.154701  0.5773503  0.5773503 -1.1547005 -0.5773503 -0.5773503
    ## 3    Hao  0.000000  0.0000000  0.0000000  0.9855986  2.1402991 -1.1547005
    ## 4   Kate  0.000000  0.0000000  0.0000000 -0.4082483 -0.9855986  0.0000000
    ## 5   Mike  0.000000 -0.5773503 -0.5773503 -0.4082483 -0.4082483  0.0000000
    ## 6   Ming  0.000000 -0.5773503 -0.5773503  0.0000000  0.0000000  1.1547005
    ## 7  Orshi -1.154701  0.5773503  0.5773503 -1.7320508 -1.1547005 -0.5773503
    ## 8   Tito -1.154701  0.0000000  0.0000000 -0.5773503 -1.1547005  0.0000000
    ##       Family    Musical      Drama  Animation     Comedy
    ## 1  0.5773503  0.5773503 -0.5773503  0.0000000  0.0000000
    ## 2  0.5773503  0.5773503 -0.5773503  0.0000000  0.0000000
    ## 3  0.9855986  0.5773503 -0.1691020  0.4082483  0.4082483
    ## 4 -0.9855986 -0.5773503 -0.4082483 -0.4082483 -0.4082483
    ## 5 -0.9855986 -0.5773503  0.1691020 -0.4082483 -0.4082483
    ## 6  0.5773503  0.5773503  0.5773503  0.0000000  0.0000000
    ## 7  0.0000000  0.0000000  0.0000000  0.0000000  0.0000000
    ## 8  0.0000000  0.0000000 -0.5773503  0.0000000  0.0000000

#### 1.1.4 Weighted Scores

Weighted scores of each movie is the dot product of vector of normalized item fetures(binary\_genre\_matrix\_nor) for the coppresonding movie and vector of IDF(binary\_IDF).

``` r
binary_IDF <- matrix(binary_IDF,1)
x <- t(binary_genre_matrix_nor)
binary_weight <- x * as.vector(binary_IDF)
binary_weight_df <- as.data.frame(binary_weight)
colnames(binary_weight_df) <- binary_genre_matrix_nor_df[,1]
binary_weight_df
```

    ##           Beauty and the Beast Guardians of the Galaxy Vol.       Logan
    ## Action               0.0000000                    0.08436706 0.08436706
    ## Crime                0.0000000                    0.00000000 0.00000000
    ## Thriller             0.0000000                    0.00000000 0.00000000
    ## Adventure            0.0000000                    0.14031808 0.00000000
    ## Fantasy              0.1403181                    0.00000000 0.00000000
    ## Sci-Fi               0.0000000                    0.31411783 0.31411783
    ## Family               0.3141178                    0.00000000 0.00000000
    ## Musical              0.4879176                    0.00000000 0.00000000
    ## Drama                0.0000000                    0.00000000 0.31411783
    ## Animation            0.0000000                    0.00000000 0.00000000
    ## Comedy               0.0000000                    0.00000000 0.00000000
    ##           Star Wars: The Last Jedi The Fate of the Furious
    ## Action                  0.08436706              0.08436706
    ## Crime                   0.00000000              0.48791758
    ## Thriller                0.00000000              0.48791758
    ## Adventure               0.14031808              0.00000000
    ## Fantasy                 0.14031808              0.00000000
    ## Sci-Fi                  0.00000000              0.00000000
    ## Family                  0.00000000              0.00000000
    ## Musical                 0.00000000              0.00000000
    ## Drama                   0.00000000              0.00000000
    ## Animation               0.00000000              0.00000000
    ## Comedy                  0.00000000              0.00000000
    ##           The Good Dinosaur Thor: Ragnarok
    ## Action           0.00000000     0.08436706
    ## Crime            0.00000000     0.00000000
    ## Thriller         0.00000000     0.00000000
    ## Adventure        0.09921987     0.14031808
    ## Fantasy          0.09921987     0.14031808
    ## Sci-Fi           0.00000000     0.00000000
    ## Family           0.22211485     0.00000000
    ## Musical          0.00000000     0.00000000
    ## Drama            0.22211485     0.00000000
    ## Animation        0.34500983     0.00000000
    ## Comedy           0.34500983     0.00000000

#### 1.1.5 Prediction

Then the dot product of the vector of the weighted scores of each movie and the vector of user-profile (binary\_rating\_matrix\_nor) for a user tell us the probability that the user will like a particular movie.

``` r
cb_binary_prediction <- binary_user_profile %*% binary_weight
colnames(cb_binary_prediction) <- colnames(binary_weight_df)
rownames(cb_binary_prediction) <-binary_user_profile_df[,1]
cb_binary_prediction
```

    ##        Beauty and the Beast Guardians of the Galaxy Vol.        Logan
    ## Alison            0.4630554                   -0.62215609 -0.64148673
    ## Eran              0.3820427                   -0.44080007 -0.46013072
    ## Hao               0.8916161                   -0.22441473 -0.41582998
    ## Kate             -0.7295907                   -0.05728462 -0.12823807
    ## Mike             -0.6485780                   -0.05728462  0.05311795
    ## Ming              0.4630554                    0.36271203  0.54406804
    ## Orshi            -0.1620254                   -0.52181275 -0.27877471
    ## Tito             -0.1620254                   -0.17843137 -0.27877471
    ##        Star Wars: The Last Jedi The Fate of the Furious The Good Dinosaur
    ## Alison               -0.2594441              0.46598000        -0.1145692
    ## Eran                 -0.3404567              0.46598000        -0.1718539
    ## Hao                   0.4386200              0.00000000         0.7732065
    ## Kate                 -0.1955819              0.00000000        -0.7295907
    ## Mike                 -0.1145692             -0.56339869        -0.5440680
    ## Ming                  0.0000000             -0.56339869         0.2564761
    ## Orshi                -0.5024821              0.46598000        -0.2864231
    ## Tito                 -0.3404567             -0.09741869        -0.3000919
    ##        Thor: Ragnarok
    ## Alison     -0.2594441
    ## Eran       -0.3404567
    ## Hao         0.4386200
    ## Kate       -0.1955819
    ## Mike       -0.1145692
    ## Ming        0.0000000
    ## Orshi      -0.5024821
    ## Tito       -0.3404567

``` r
image(cb_binary_prediction, main = "Probability", xlab = "Movie", ylab = "Friend")
```

![](DATA643_project2_files/figure-markdown_github/unnamed-chunk-12-1.png)

From the content-based prediction, it seems that Beauty and the Beast is the most popular movie among these 7 movies of my friends. Orshi tend to give negative ratings to all movies except The Fate of the Furious and this movie is the only one get positive predition for Orshi. I know that Kate who only watched Beauty and the Beast and The Good Dinosaur and she likes both movies. But the prediction suggests that Kate would not like any of these movies.

#### 1.2 Non-binary representation

#### 1.2.1 Feature extraction

The storyline will be use to extract the features for each movie

``` r
movie_1 <- data.frame(lapply(movie, as.character), stringsAsFactors=FALSE)
movie_words <- movie_1[,c('Title','storyline')]
movie_words <- as_data_frame(movie_words)

# frequency of each words in each movie
movie_words <- unnest_tokens(movie_words,word, storyline) %>% 
  count( Title, word, sort = TRUE) %>% 
  ungroup()

# count total words in each movie
total_words <- movie_words %>% group_by(Title) %>% summarize(total = sum(n))

(movie_words <- left_join(movie_words, total_words))
```

    ## Joining, by = "Title"

    ## # A tibble: 448 × 4
    ##                             Title  word     n total
    ##                             <chr> <chr> <int> <int>
    ## 1         The Fate of the Furious   the    13   124
    ## 2                           Logan   the    11   206
    ## 3                           Logan    to    11   206
    ## 4                           Logan     a     8   206
    ## 5         The Fate of the Furious   and     7   124
    ## 6  Guardians of the Galaxy Vol. 2   the     6    63
    ## 7         The Fate of the Furious    of     6   124
    ## 8               The Good Dinosaur   the     6    68
    ## 9                  Thor: Ragnarok   the     6    49
    ## 10                          Logan   and     5   206
    ## # ... with 438 more rows

``` r
ggplot(movie_words, aes(n/total, fill = Title)) +
  geom_histogram(show.legend = FALSE) +
  facet_wrap(~Title, ncol = 2, scales = "free_y")
```

    ## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.

![](DATA643_project2_files/figure-markdown_github/unnamed-chunk-14-1.png)

The above figures exhibit similar distributions for all the movies. From the pattern of distribution,we can see that only a few of words that occur frequently. From here TF-IDF will be used to find the important words of each movie by using bind\_tf\_idf function in tidytext package.

``` r
# calculate TF-IDF 
(movie_words <- movie_words %>%
  bind_tf_idf(word, Title, n)) %>% 
  arrange(desc(tf_idf))
```

    ## # A tibble: 448 × 7
    ##                             Title      word     n total         tf
    ##                             <chr>     <chr> <int> <int>      <dbl>
    ## 1            Beauty and the Beast      only     3    58 0.05172414
    ## 2        Star Wars: The Last Jedi       her     2    34 0.05882353
    ## 3            Beauty and the Beast        be     2    58 0.03448276
    ## 4            Beauty and the Beast      form     2    58 0.03448276
    ## 5  Guardians of the Galaxy Vol. 2 guardians     2    63 0.03174603
    ## 6               The Good Dinosaur      arlo     2    68 0.02941176
    ## 7               The Good Dinosaur dinosaurs     2    68 0.02941176
    ## 8        Star Wars: The Last Jedi      2015     1    34 0.02941176
    ## 9        Star Wars: The Last Jedi   awakens     1    34 0.02941176
    ## 10       Star Wars: The Last Jedi   chapter     1    34 0.02941176
    ## # ... with 438 more rows, and 2 more variables: idf <dbl>, tf_idf <dbl>

#### 1.2.2 Pridiction

``` r
# find unique word
n_occur <- data.frame(table(movie_words$word))
# filter words occur at least in two movies
movie_words_dup <- movie_words[movie_words$word %in% n_occur$Var1[n_occur$Freq > 1],]

# the tf-idf of each word for each movie
imt <- spread(movie_words_dup, word,tf_idf, fill = NA, convert = FALSE, drop = TRUE,sep = NULL)
imt <- imt[-c(2:5)]
feature <-apply(imt[,c(2:63)], 2, function(x) tapply(x, imt$Title, sum,na.rm=T))

#find the similarity
feature_m <- as(feature,"realRatingMatrix")
(cos_sim <- similarity(feature_m, method = 'cosine', which = 'item'))
```

    ##                                Beauty and the Beast
    ## Guardians of the Galaxy Vol. 2           0.13699624
    ## Logan                                    0.27962005
    ## Star Wars: The Last Jedi                 0.09548751
    ## The Fate of the Furious                  0.11306317
    ## The Good Dinosaur                        0.24560655
    ## Thor: Ragnarok                           0.25932317
    ##                                Guardians of the Galaxy Vol. 2      Logan
    ## Guardians of the Galaxy Vol. 2                                          
    ## Logan                                              0.30845922           
    ## Star Wars: The Last Jedi                           0.03810912 0.31202199
    ## The Fate of the Furious                            0.50500573 0.36920602
    ## The Good Dinosaur                                  0.04136403 0.27563527
    ## Thor: Ragnarok                                     0.17676893 0.33536068
    ##                                Star Wars: The Last Jedi
    ## Guardians of the Galaxy Vol. 2                         
    ## Logan                                                  
    ## Star Wars: The Last Jedi                               
    ## The Fate of the Furious                      0.09696210
    ## The Good Dinosaur                            0.21336818
    ## Thor: Ragnarok                               0.02092179
    ##                                The Fate of the Furious The Good Dinosaur
    ## Guardians of the Galaxy Vol. 2                                          
    ## Logan                                                                   
    ## Star Wars: The Last Jedi                                                
    ## The Fate of the Furious                                                 
    ## The Good Dinosaur                           0.20825389                  
    ## Thor: Ragnarok                              0.17437285        0.06572412

From the similarity results, we can see the most silimar movie for each movie, for example, Beauty and the Beast and Logan are most similar.Base on these results, we can recommend Logan to Aliaon, Eran, Hao, and Ming as they gave postive ratign to Beauty and the Beast based on the nomalized binary rating table.

#### **2. Collaborative Filtering**

#### 2.1 Coercion the Data to Rating Matrices

``` r
cf_rating_matrix <- binary_rating_matrix [-c(1)]
cf_rating_matrix <- as.matrix(cf_rating_matrix)
cf_rating <- as(cf_rating_matrix, "realRatingMatrix")
getRatingMatrix(cf_rating)
```

    ## 8 x 7 sparse Matrix of class "dgCMatrix"
    ##   Beauty and the Beast Guardians of the Galaxy Vol.  Logan
    ## 1                    5                           2.0   1.0
    ## 2                    5                           .     1.5
    ## 3                    4                           3.0   1.5
    ## 4                    5                           .     .  
    ## 5                    2                           3.0   4.0
    ## 6                    4                           4.0   5.0
    ## 7                    .                           1.5   .  
    ## 8                    .                           4.5   4.0
    ##   Star Wars: The Last Jedi The Fate of the Furious The Good Dinosaur
    ## 1                      1.5                     4.5                 .
    ## 2                      1.0                     3.0                 .
    ## 3                      4.5                     .                   5
    ## 4                      .                       .                   5
    ## 5                      5.0                     2.0                 2
    ## 6                      .                       2.0                 .
    ## 7                      2.0                     2.5                 .
    ## 8                      4.0                     .                   .
    ##   Thor: Ragnarok
    ## 1            .  
    ## 2            2.0
    ## 3            4.0
    ## 4            .  
    ## 5            .  
    ## 6            1.5
    ## 7            2.0
    ## 8            4.0

``` r
# identical(as(cf_rating, "matrix"),cf_rating_matrix) 
# TRUE

# rowCounts(cf_rating[1,])
# as(cf_rating[1,],'list')
# rowMeans(cf_rating[1,])
hist(getRatings(cf_rating),breaks = 10)
```

![](DATA643_project2_files/figure-markdown_github/unnamed-chunk-17-1.png)

``` r
# row centering normalization
hist(getRatings(normalize(cf_rating)), breaks=8)
```

![](DATA643_project2_files/figure-markdown_github/unnamed-chunk-17-2.png)

``` r
# Z-score normalization
hist(getRatings(normalize(cf_rating, method="Z-score")), breaks=8)
```

![](DATA643_project2_files/figure-markdown_github/unnamed-chunk-17-3.png)

``` r
# the mean rating for each movie
hist(colMeans(cf_rating), breaks=10)
```

![](DATA643_project2_files/figure-markdown_github/unnamed-chunk-17-4.png)

The distribution of ratings is nearly a normal distribution. The distribution of means of ratings is not a normal distribution.

#### 2.2 Normalization

``` r
cf_rating_nor <- normalize(cf_rating)
image(cf_rating, main = "Raw Ratings")
```

![](DATA643_project2_files/figure-markdown_github/unnamed-chunk-18-1.png)

``` r
image(cf_rating_nor, main = "Normalized Ratings")
```

![](DATA643_project2_files/figure-markdown_github/unnamed-chunk-18-2.png)

#### 2.3 IBCF: Item-Based Collaborative Filtering

``` r
# Creation of the model
cf_train <- as(cf_rating_matrix[1:5,],"realRatingMatrix")
cf_test <- as(cf_rating_matrix[6:8,],"realRatingMatrix")
cf_item_r <- Recommender(cf_train, method = "IBCF")

# Making predictions
cf_item_recom <- predict(cf_item_r, cf_test, type="ratings")
as(cf_item_recom, "matrix")
```

    ##   Beauty and the Beast Guardians of the Galaxy Vol.     Logan
    ## 6                   NA                            NA       NA
    ## 7             2.107320                            NA 1.947241
    ## 8             4.103111                            NA       NA
    ##   Star Wars: The Last Jedi The Fate of the Furious The Good Dinosaur
    ## 6                 3.049213                      NA          3.571779
    ## 7                       NA                      NA          1.865393
    ## 8                       NA                 4.13445          4.136376
    ##   Thor: Ragnarok
    ## 6             NA
    ## 7             NA
    ## 8             NA

``` r
# Compare to the real rating, only not rated cells have been predicted
cf_rating_matrix[6:8,]
```

    ##   Beauty and the Beast Guardians of the Galaxy Vol.  Logan
    ## 6                    4                           4.0     5
    ## 7                   NA                           1.5    NA
    ## 8                   NA                           4.5     4
    ##   Star Wars: The Last Jedi The Fate of the Furious The Good Dinosaur
    ## 6                       NA                     2.0                NA
    ## 7                        2                     2.5                NA
    ## 8                        4                      NA                NA
    ##   Thor: Ragnarok
    ## 6            1.5
    ## 7            2.0
    ## 8            4.0

``` r
# similarity table
(cos_sim <- similarity(cf_train, method = 'cosine', which = 'item'))
```

    ##           1         2         3         4
    ## 2 0.8917774                              
    ## 3 0.5024690 0.5751369                    
    ## 4 0.4879500 0.5504819 0.6764814          
    ## 5 0.6397604 0.5338951 0.7492478 0.3592106

``` r
image(as.matrix(cos_sim), main = "Item Similarity",xlab="Movie",ylab="Movie")
```

![](DATA643_project2_files/figure-markdown_github/unnamed-chunk-21-1.png)

From the similarity table we can find the most similar movie for each movie, for example, movie-1's neighbour is movie-2 and movie-2's neighbour is movie-6 in training set.

#### 2.4 UBCF: User-Based Collaborative Filtering

``` r
# Creation of the model
cf_user_r <- Recommender(cf_train, method = "UBCF")

# Making predictions
cf_user_recom <- predict(cf_user_r, cf_test, type="ratings")
as(cf_user_recom, "matrix")
```

    ##   Beauty and the Beast Guardians of the Galaxy Vol.  Logan
    ## 6                   NA                            NA    NA
    ## 7                   NA                            NA    NA
    ## 8                   NA                            NA    NA
    ##   Star Wars: The Last Jedi The Fate of the Furious The Good Dinosaur
    ## 6                 3.651425                      NA          3.661947
    ## 7                       NA                      NA                NA
    ## 8                       NA                      NA                NA
    ##   Thor: Ragnarok
    ## 6             NA
    ## 7             NA
    ## 8             NA

``` r
cf_recom_list <- as(cf_user_recom, "list") #convert recommenderlab object to readable list
```

``` r
# similarity table
(cos_sim <- similarity(cf_train, method = 'cosine', which = 'user'))
```

    ##           1         2         3         4
    ## 2 0.8917774                              
    ## 3 0.5024690 0.5751369                    
    ## 4 0.4879500 0.5504819 0.6764814          
    ## 5 0.6397604 0.5338951 0.7492478 0.3592106

``` r
image(as.matrix(cos_sim), main = "Item Similarity",xlab = "Friend", ylab = "Friend")
```

![](DATA643_project2_files/figure-markdown_github/unnamed-chunk-23-1.png)

#### 2.5 Evaluation of Predicted Ratings

``` r
cf_itm_e <- evaluationScheme(cf_rating, method="split", train=0.8, given=2, goodRating = 2.5)
r1 <- Recommender(getData(cf_itm_e, "train"), "IBCF")
r2 <- Recommender(getData(cf_itm_e, "train"), "UBCF")
p1 <- predict(r1, getData(cf_itm_e, "known"), type="ratings")
p2 <- predict(r2, getData(cf_itm_e, "known"), type="ratings")
(error <- rbind(UBCF = calcPredictionAccuracy(p1, getData(cf_itm_e, "unknown")),IBCF = calcPredictionAccuracy(p2, getData(cf_itm_e, "unknown"))))
```

    ##      RMSE MSE MAE
    ## UBCF  NaN NaN NaN
    ## IBCF  NaN NaN NaN

Item-base callborative filtering performed better than user-base callborative filtering as RMSE of IBSF is lower than that of UBCF.

#### 2.6 Evaluation of a top-N Recommender Algorithm

``` r
set.seed(2002)
scheme <- evaluationScheme(cf_train, method="cross", k=4, given=2,goodRating=2.5)
results <- evaluate(scheme, method="IBCF", type = "topNList")
```

    ## IBCF run fold/sample [model time/prediction time]
    ##   1  [0.01sec/0sec] 
    ##   2  [0sec/0sec] 
    ##   3  [0sec/0.01sec] 
    ##   4  [0sec/0.02sec]

``` r
rslt <- getConfusionMatrix(results)[[1]]
avg(results)
```

    ##       TP    FP    FN    TN precision    recall       TPR       FPR
    ## 1  0.250 0.625 1.375 2.750 0.2500000 0.1666667 0.1666667 0.1979167
    ## 2  0.625 1.125 1.000 2.250 0.3125000 0.4166667 0.4166667 0.3750000
    ## 3  1.250 1.375 0.375 2.000 0.4583333 0.7708333 0.7708333 0.4375000
    ## 4  1.250 2.250 0.375 1.125 0.3437500 0.7708333 0.7708333 0.7291667
    ## 5  1.500 2.500 0.125 0.875 0.3875000 0.9583333 0.9583333 0.7916667
    ## 6  1.500 2.500 0.125 0.875 0.3875000 0.9583333 0.9583333 0.7916667
    ## 7  1.500 2.500 0.125 0.875 0.3875000 0.9583333 0.9583333 0.7916667
    ## 8  1.500 2.500 0.125 0.875 0.3875000 0.9583333 0.9583333 0.7916667
    ## 9  1.500 2.500 0.125 0.875 0.3875000 0.9583333 0.9583333 0.7916667
    ## 10 1.500 2.500 0.125 0.875 0.3875000 0.9583333 0.9583333 0.7916667

``` r
plot(results, annotate=TRUE, main = "ROC curve for recommender method IBCF")
```

![](DATA643_project2_files/figure-markdown_github/unnamed-chunk-25-1.png)

``` r
plot(results, "prec/rec", annotate=TRUE, ylim=c(0,max(rslt[,'precision']/rslt[,'recall'])))
```

![](DATA643_project2_files/figure-markdown_github/unnamed-chunk-25-2.png)

*Referecnce:*

1.  Shuvayan Das,2015,Analytics Vidhya, Beginners Guide to learn about Content Based Recommender Engines. <https://www.analyticsvidhya.com/blog/2015/08/beginners-guide-learn-content-based-recommender-systems/>

2.  Julia Silge and David Robinson, 2017, Term Frequency and Inverse Document Frequency (tf-idf) Using Tidy Data. Principles. <https://cran.r-project.org/web/packages/tidytext/vignettes/tf_idf.html>
