DATA643\_Project5\_Spark
================
Yun Mai
July 8, 2017

``` r
install.packages("sparklyr",repos = "http://cran.us.r-project.org")
packageVersion("sparklyr")
```

``` r
require(sparklyr)
```

    ## Loading required package: sparklyr

``` r
# tried to install the version 2.1.0
# sparklyr::spark_install(version = "2.1.0")
# But got a error message:
# Error in spark_install_find(version, hadoop_version, installedOnly = FALSE,  : Spark version not available.

# checke the spark version available for installation from sparklyr 
# spark_available_versions()
# Error in file(file, "rt") : cannot open the connection

# there is no 2.1.0 so chose spark 2.0.2, hadoop: 2.7
spark_install(version = "2.0.2", hadoop_version = 2.7, reset = TRUE, logging = "INFO", verbose = interactive())

# Installing Spark 2.0.2 for Hadoop 2.7 or later.
# Downloading from:- 'https://d3kbcqa49mib13.cloudfront.net/spark-2.0.2-bin-hadoop2.7.tgz'
# Installing to:- 'C:\Users\lzq\AppData\Local\rstudio\spark\Cache/spark-2.0.2-bin-hadoop2.7'
# trying URL 'https://d3kbcqa49mib13.cloudfront.net/spark-2.0.2-bin-hadoop2.7.tgz'
# Content type 'application/x-tar' length 187426587 bytes (178.7 MB)
# downloaded 178.7 MB

spark_installed_versions()
```

    ##   spark hadoop                       dir
    ## 1 2.0.2    2.7 spark-2.0.2-bin-hadoop2.7
    ## 2 2.1.0    2.7 spark-2.1.0-bin-hadoop2.7

``` r
# so in addition to Spark 2.0.2, there are 2.1.0 which installed before under the directory "C:\Users\lzq\AppData\Local\rstudio\spark\Cache"

# devtools::install_github("rstudio/sparklyr") 
# if install sparklyr from rstudio github, we will get newer version of sparklyr and Spark (such as 2.1.0) 
```

``` r
# check the current SPARK_HOME
Sys.getenv("SPARK_HOME")
```

    ## [1] "D:\\spark-2.1.1-bin-hadoop2.7\\bin"

``` r
#check config
spark_config()
```

    ## $sparklyr.cores.local
    ## [1] 4
    ## 
    ## $spark.sql.shuffle.partitions.local
    ## [1] 4
    ## 
    ## $spark.env.SPARK_LOCAL_IP.local
    ## [1] "127.0.0.1"
    ## 
    ## $sparklyr.csv.embedded
    ## [1] "^1.*"
    ## 
    ## $`sparklyr.shell.driver-class-path`
    ## [1] ""
    ## 
    ## attr(,"config")
    ## [1] "default"
    ## attr(,"file")
    ## [1] "D:\\R\\R-3.4.1\\library\\sparklyr\\conf\\config-template.yml"

``` r
#change SPARK_HOME
Sys.setenv(SPARK_HOME="C:/Users/lzq/AppData/Local/rstudio/spark/Cache/spark-2.0.2-bin-hadoop2.7")

# connect to spark
sc <- spark_connect(master = "local",version ="2.0.2")

# succeed ! Got the following message
# Created default hadoop bin directory under: C:\Users\lzq\AppData\Local\rstudio\spark\Cache\spark-2.0.2-bin-hadoop2.7\tmp\hadoop
```

``` r
# verify the spark home directory
spark_home_dir()
```

    ## [1] "C:\\Users\\lzq\\AppData\\Local\\rstudio\\spark\\Cache/spark-2.0.2-bin-hadoop2.7"

To verify the connection

``` r
#iris_tbl <- copy_to(sc, iris)
#iris_tbl
```

I put a lot of efforts to make sparklyr work in Windows 10 but failed to wirte data into spark. I posted the issue in RStudion/sparklyr for help but did not get response yet. So I could not finish project 5 in time. I will try SparkR. Hopefully I can figure out how to use Spark in R and apply it in the final project.
