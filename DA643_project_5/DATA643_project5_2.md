DATA643\_Project5\_Spark
================
Yun Mai
July 8, 2017

### 1. Goal

Adapt one of your recommendation systems to work with Apache Spark and compare the performance with your previous iteration. Consider the efficiency of the system and the added complexity of using Spark. I will use sparklyr for this project.

The question to be answered: For your given recommender system's data, algorithm(s), and (envisioned) implementation, at what point would you see moving to a distributed platform such as Spark becoming necessary?

### 2. Installing sparklyr and Spark and Loading the Data

``` r
install.packages("sparklyr", repos="http://cran.rstudio.com/")
```

    ## package 'sparklyr' successfully unpacked and MD5 sums checked

    ## Warning: cannot remove prior installation of package 'sparklyr'

    ## 
    ## The downloaded binary packages are in
    ##  C:\Users\lzq\AppData\Local\Temp\RtmpaasW8t\downloaded_packages

**Install a local version of Spark for development purposes:**

``` r
library(sparklyr)
spark_install(version = "2.0.2")
```

\*\*To upgrade to the latest version of sparklyr, run the following command and restart R <session:**>

``` r
library(dplyr)
```

    ## 
    ## Attaching package: 'dplyr'

    ## The following objects are masked from 'package:stats':
    ## 
    ##     filter, lag

    ## The following objects are masked from 'package:base':
    ## 
    ##     intersect, setdiff, setequal, union

``` r
devtools::install_github("rstudio/sparklyr")
```

    ## Downloading GitHub repo rstudio/sparklyr@master
    ## from URL https://api.github.com/repos/rstudio/sparklyr/zipball/master

    ## Installing sparklyr

    ## "D:/R/R-34~1.1/bin/x64/R" --no-site-file --no-environ --no-save  \
    ##   --no-restore --quiet CMD INSTALL  \
    ##   "C:/Users/lzq/AppData/Local/Temp/RtmpaasW8t/devtools32e834a8274c/rstudio-sparklyr-ee4127f"  \
    ##   --library="D:/R/R-3.4.1/library" --install-tests

    ## 

**Connect to a local instance of Spark via the spark\_connect function:**

``` r
library(sparklyr)
```

    ## 
    ## Attaching package: 'sparklyr'

    ## The following object is masked from 'package:dplyr':
    ## 
    ##     top_n

``` r
# Change SPARK_HOME and JAVA_HOME to accommodate sparklyr as they might be set as different directory for scela and SparkR
Sys.setenv(JAVA_HOME = "C:/Java/jre1.8.0_131")
Sys.setenv(SPARK_HOME = "C:/Users/lzq/AppData/Local/rstudio/spark/Cache/spark-2.0.2-bin-hadoop2.7")

sc <- spark_connect(master = "local")
```

The returned Spark connection (sc) provides a remote dplyr data source to the Spark cluster.

\*\* To verify the connection\*\*

``` r
library(dplyr)
iris <- iris
iris_tbl <- copy_to(sc, iris)
```

### 3. Building a Recommender System using Singular Value Decompositionon with recommenderlab

#### 3.1 Load the packages and data

``` r
suppressWarnings(suppressMessages(library(recommenderlab)))
suppressWarnings(suppressMessages(library(knitr)))
suppressWarnings(suppressMessages(library(tidyr)))
suppressWarnings(suppressMessages(library(ggplot2)))
data(MovieLense)
```

#### 3.2 Basic model

``` r
# check if there is abnormal ratings in the data
table(MovieLense@data@x[] > 5)
```

    ## 
    ## FALSE 
    ## 99392

``` r
table(MovieLense@data@x[] < 1)
```

    ## 
    ## FALSE 
    ## 99392

``` r
######################### SVD ############################

# Create and maintain evaluation schemes; divide the data into 90% training 10% test
div <- evaluationScheme(MovieLense, method="split", train = 0.9, k=10, given = 15, goodRating = 3)
div
```

    ## Evaluation scheme with 15 items given
    ## Method: 'split' with 10 run(s).
    ## Training set proportion: 0.900
    ## Good ratings: >=3.000000
    ## Data set: 943 x 1664 rating matrix of class 'realRatingMatrix' with 99392 ratings.

``` r
# Create the recommender based on SVD algorithm using the training data
r.svd <- Recommender(getData(div, "train"), "SVD", parameter = list(k=50, maxiter = 100, normalize = "Z-score"))

# Compute predicted ratings for test data that is known using the UBCF algorithm
p.svd <- predict(r.svd, getData(div, "known"), type = "ratings")
# Created evaluation scheme to evaluate the recommender method SVD
results <- evaluate(div, method="SVD", type = "topNList", n=c(1,3,5,10,15,20))
```

    ## SVD run fold/sample [model time/prediction time]
    ##   1  [1.19sec/0.28sec] 
    ##   2  [0.43sec/0.3sec] 
    ##   3  [0.27sec/0.44sec] 
    ##   4  [0.22sec/0.29sec] 
    ##   5  [0.43sec/0.26sec] 
    ##   6  [0.22sec/0.45sec] 
    ##   7  [0.23sec/0.26sec] 
    ##   8  [0.44sec/0.24sec] 
    ##   9  [0.42sec/0.26sec] 
    ##   10  [0.25sec/0.25sec]

``` r
# Show the top 6 movies for 6 users
getRatingMatrix(p.svd)[1:6,1:6]
```

    ## 6 x 6 sparse Matrix of class "dgCMatrix"
    ##    Toy Story (1995) GoldenEye (1995) Four Rooms (1995) Get Shorty (1995)
    ## 7          4.137323         4.151321          4.180323          4.234680
    ## 19         3.585767         3.673531          3.625392          .       
    ## 27         3.108214         3.091353          3.077586          3.211699
    ## 31         3.872994         3.896926          3.850274          3.905186
    ## 40         2.305568         2.343640          2.361140          2.392548
    ## 41         3.764588         3.656159          3.656890          3.654702
    ##    Copycat (1995) Shanghai Triad (Yao a yao yao dao waipo qiao) (1995)
    ## 7        4.212309                                             4.214174
    ## 19       3.635150                                             3.657766
    ## 27       3.107028                                             3.144963
    ## 31       3.815304                                             3.869124
    ## 40       2.327163                                             2.337902
    ## 41       3.667307                                             3.660873

``` r
# Calculate the error between training prediction and unknown test data
error <- data.frame(SVD = calcPredictionAccuracy(p.svd, getData(div, "unknown")))
kable(error)
```

|      |        SVD|
|------|----------:|
| RMSE |  1.0491261|
| MSE  |  1.1006656|
| MAE  |  0.8438333|

Next, I will use machine learning functuion in spraklyr to build the recommendtions system. There is no svd algorithm in spraklyr. But Principal Component Analysis (PCA) is a simple application of SVD, which is availabe in spraklyr machine learning fuction, so I will use PCA.

### 4. Building a Recommender System under Spark environment

Principal Component Analysis (PCA)

``` r
movie_df <- as(MovieLense, 'data.frame')
movie_df$user <- sapply(movie_df$user,function(x) as.numeric(as.character(x)))
movie_df$item  <- sapply(movie_df$item,function(x) as.character(x))
movie_mx <- spread(movie_df, item, rating)
movie_mx$user <- sapply(movie_mx$user,function(x) as.numeric(x))
movie_mx[is.na(movie_mx)]<- 0

#copy data to spark
movie_tbl <- sdf_copy_to(sc,movie_mx, "movie_DF", overwrite=T)

movies <- paste(colnames(movie_mx)[-1])

pca_model <- ml_pca(movie_tbl,features = paste(colnames(movie_tbl)[2:51]))
```

    ## * No rows dropped by 'na.omit' call

``` r
pca_df <-as.data.frame(pca_model$components)

suppressWarnings(suppressMessages(library(tibble)))
pca_df <-rownames_to_column(pca_df,var = "title")
head(pca_df[,1:6])
```

    ##                       title           PC1          PC2          PC3
    ## 1    Til_There_Was_You_1997 -0.0045006726  0.009503177  0.005508996
    ## 2                 1900_1994 -0.0013667635 -0.005286611 -0.001096932
    ## 3       101_Dalmatians_1996 -0.0560940728  0.038000814 -0.056204477
    ## 4         12_Angry_Men_1957 -0.1998470765 -0.139357050  0.337153489
    ## 5                  187_1997 -0.0005624931  0.060056956  0.001314487
    ## 6 2_Days_in_the_Valley_1996 -0.0525096026  0.067191572 -0.035024667
    ##            PC4          PC5
    ## 1  0.013649136  0.015251819
    ## 2 -0.002857479 -0.004839243
    ## 3  0.243564077 -0.002658952
    ## 4  0.033926421  0.494198469
    ## 5  0.027392024  0.023169036
    ## 6  0.076000865  0.047623924

``` r
ggplot(pca_df, aes(x = PC1, y = PC2, color = title, label = title)) +
  geom_point(size = 2, alpha = 0.6) +
  labs(title = "Where the Movies Fall on the First Two Principal Components", x = paste0("PC1: ", round(pca_model$explained.variance[1], digits = 2) * 100, "% variance"),y = paste0("PC2: ", round(pca_model$explained.variance[2], digits = 2) * 100, "% variance")) +
  guides(fill = FALSE, color = FALSE)
```

![](DATA643_project5_files/figure-markdown_github-ascii_identifiers/unnamed-chunk-9-1.png)

``` r
# movie_df_l <- sdf_copy_to(sc,movie_df, "movie_DF_long", overwrite=T)
# Partition
# model_data <- tbl(sc, 'movie_DF_long') 
# partitions <- model_data %>%
#  sdf_partition(train = .9, test = .1)
# pca_model <- ml_pca(partitions$train,features = paste(colnames(model_data)[2:51]))
# Predict on test set
# predicts <- sdf_predict(pca_model, partitions$test) 
```

It took a fairly long time to run PCA algorithm. And I can not compute based on all the movies as it ran out of memory and gave me a error message. So I selected the first 50 movies to for the model. I guess using Spark will be beneficial for other algorithms such as ALS, Kmean, etc.

Reference:

<https://rpubs.com/Thong/data-analysis-with-r-and-spark> (Thank for this post.It helped me going to the right direction in setting up sparklyr and get sparklyr to work.)
