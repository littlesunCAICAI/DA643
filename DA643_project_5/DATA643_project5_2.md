DATA643\_Project5\_Spark
================
Yun Mai
July 8, 2017

#### Goal

Adapt one of your recommendation systems to work with Apache Spark and compare the performance with your previous iteration. Consider the efficiency of the system and the added complexity of using Spark. I want to use sparklyr for this project.

The question to be answered: For your given recommender system's data, algorithm(s), and (envisioned) implementation, at what point would you see moving to a distributed platform such as Spark becoming necessary?

### Set up sparklyr

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

#### To verify the connection

``` r
#iris_tbl <- copy_to(sc, iris)
#iris_tbl
```

### Troubleshooting

``` r
spark_log(sc)
```

    ## 17/07/11 10:17:15 INFO NettyBlockTransferService: Server created on 192.168.1.151:63513
    ## 17/07/11 10:17:15 INFO BlockManagerMaster: Registering BlockManager BlockManagerId(driver, 192.168.1.151, 63513)
    ## 17/07/11 10:17:15 INFO BlockManagerMasterEndpoint: Registering block manager 192.168.1.151:63513 with 413.9 MB RAM, BlockManagerId(driver, 192.168.1.151, 63513)
    ## 17/07/11 10:17:15 INFO BlockManagerMaster: Registered BlockManager BlockManagerId(driver, 192.168.1.151, 63513)
    ## 17/07/11 10:17:15 WARN SparkContext: Use an existing SparkContext, some configuration may not take effect.
    ## 17/07/11 10:17:15 WARN SparkContext: Use an existing SparkContext, some configuration may not take effect.
    ## 17/07/11 10:17:15 INFO HiveSharedState: Warehouse path is 'C:UserslzqAppDataLocal
    ## studiosparkCachespark-2.0.2-bin-hadoop2.7    mphive'.
    ## 17/07/11 10:17:17 INFO SparkContext: Invoking stop() from shutdown hook
    ## 17/07/11 10:17:17 INFO SparkUI: Stopped Spark web UI at http://192.168.1.151:4041
    ## 17/07/11 10:17:17 INFO MapOutputTrackerMasterEndpoint: MapOutputTrackerMasterEndpoint stopped!
    ## 17/07/11 10:17:17 INFO MemoryStore: MemoryStore cleared
    ## 17/07/11 10:17:17 INFO BlockManager: BlockManager stopped
    ## 17/07/11 10:17:17 INFO BlockManagerMaster: BlockManagerMaster stopped
    ## 17/07/11 10:17:17 INFO OutputCommitCoordinator$OutputCommitCoordinatorEndpoint: OutputCommitCoordinator stopped!
    ## 17/07/11 10:17:17 INFO SparkContext: Successfully stopped SparkContext
    ## 17/07/11 10:17:17 INFO ShutdownHookManager: Shutdown hook called
    ## 17/07/11 10:17:17 INFO ShutdownHookManager: Deleting directory C:\Users\lzq\AppData\Local\Temp\spark-ae9c7485-3a82-49b1-8ddb-6df475d34c6f
    ## 17/07/11 10:17:19 INFO SparkContext: Invoking stop() from shutdown hook
    ## 17/07/11 10:17:19 INFO SparkUI: Stopped Spark web UI at http://127.0.0.1:4040
    ## 17/07/11 10:17:19 INFO MapOutputTrackerMasterEndpoint: MapOutputTrackerMasterEndpoint stopped!
    ## 17/07/11 10:17:19 INFO MemoryStore: MemoryStore cleared
    ## 17/07/11 10:17:19 INFO BlockManager: BlockManager stopped
    ## 17/07/11 10:17:19 INFO BlockManagerMaster: BlockManagerMaster stopped
    ## 17/07/11 10:17:19 INFO OutputCommitCoordinator$OutputCommitCoordinatorEndpoint: OutputCommitCoordinator stopped!
    ## 17/07/11 10:17:19 INFO SparkContext: Successfully stopped SparkContext
    ## 17/07/11 10:17:19 INFO ShutdownHookManager: Shutdown hook called
    ## 17/07/11 10:17:19 INFO ShutdownHookManager: Deleting directory C:\Users\lzq\AppData\Local\Temp\spark-2fe40a61-3bc3-460e-b5b2-9fc03c46face
    ## 17/07/11 10:18:41 INFO SparkContext: Running Spark version 2.0.2
    ## 17/07/11 10:18:41 ERROR Shell: Failed to locate the winutils binary in the hadoop binary path
    ## java.io.IOException: Could not locate executable C:\Users\lzq\AppData\Local\rstudio\spark\Cache\spark-2.0.2-bin-hadoop2.7\tmp\hadoop\bin\bin\winutils.exe in the Hadoop binaries.
    ##  at org.apache.hadoop.util.Shell.getQualifiedBinPath(Shell.java:379)
    ##  at org.apache.hadoop.util.Shell.getWinUtilsPath(Shell.java:394)
    ##  at org.apache.hadoop.util.Shell.<clinit>(Shell.java:387)
    ##  at org.apache.hadoop.util.StringUtils.<clinit>(StringUtils.java:80)
    ##  at org.apache.hadoop.security.SecurityUtil.getAuthenticationMethod(SecurityUtil.java:611)
    ##  at org.apache.hadoop.security.UserGroupInformation.initialize(UserGroupInformation.java:273)
    ##  at org.apache.hadoop.security.UserGroupInformation.ensureInitialized(UserGroupInformation.java:261)
    ##  at org.apache.hadoop.security.UserGroupInformation.loginUserFromSubject(UserGroupInformation.java:791)
    ##  at org.apache.hadoop.security.UserGroupInformation.getLoginUser(UserGroupInformation.java:761)
    ##  at org.apache.hadoop.security.UserGroupInformation.getCurrentUser(UserGroupInformation.java:634)
    ##  at org.apache.spark.util.Utils$$anonfun$getCurrentUserName$1.apply(Utils.scala:2345)
    ##  at org.apache.spark.util.Utils$$anonfun$getCurrentUserName$1.apply(Utils.scala:2345)
    ##  at scala.Option.getOrElse(Option.scala:121)
    ##  at org.apache.spark.util.Utils$.getCurrentUserName(Utils.scala:2345)
    ##  at org.apache.spark.SparkContext.<init>(SparkContext.scala:294)
    ##  at org.apache.spark.SparkContext$.getOrCreate(SparkContext.scala:2258)
    ##  at org.apache.spark.SparkContext.getOrCreate(SparkContext.scala)
    ##  at sun.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
    ##  at sun.reflect.NativeMethodAccessorImpl.invoke(Unknown Source)
    ##  at sun.reflect.DelegatingMethodAccessorImpl.invoke(Unknown Source)
    ##  at java.lang.reflect.Method.invoke(Unknown Source)
    ##  at sparklyr.Invoke$.invoke(invoke.scala:94)
    ##  at sparklyr.StreamHandler$.handleMethodCall(stream.scala:89)
    ##  at sparklyr.StreamHandler$.read(stream.scala:55)
    ##  at sparklyr.BackendHandler.channelRead0(handler.scala:49)
    ##  at sparklyr.BackendHandler.channelRead0(handler.scala:14)
    ##  at io.netty.channel.SimpleChannelInboundHandler.channelRead(SimpleChannelInboundHandler.java:105)
    ##  at io.netty.channel.AbstractChannelHandlerContext.invokeChannelRead(AbstractChannelHandlerContext.java:308)
    ##  at io.netty.channel.AbstractChannelHandlerContext.fireChannelRead(AbstractChannelHandlerContext.java:294)
    ##  at io.netty.handler.codec.MessageToMessageDecoder.channelRead(MessageToMessageDecoder.java:103)
    ##  at io.netty.channel.AbstractChannelHandlerContext.invokeChannelRead(AbstractChannelHandlerContext.java:308)
    ##  at io.netty.channel.AbstractChannelHandlerContext.fireChannelRead(AbstractChannelHandlerContext.java:294)
    ##  at io.netty.handler.codec.ByteToMessageDecoder.channelRead(ByteToMessageDecoder.java:244)
    ##  at io.netty.channel.AbstractChannelHandlerContext.invokeChannelRead(AbstractChannelHandlerContext.java:308)
    ##  at io.netty.channel.AbstractChannelHandlerContext.fireChannelRead(AbstractChannelHandlerContext.java:294)
    ##  at io.netty.channel.DefaultChannelPipeline.fireChannelRead(DefaultChannelPipeline.java:846)
    ##  at io.netty.channel.nio.AbstractNioByteChannel$NioByteUnsafe.read(AbstractNioByteChannel.java:131)
    ##  at io.netty.channel.nio.NioEventLoop.processSelectedKey(NioEventLoop.java:511)
    ##  at io.netty.channel.nio.NioEventLoop.processSelectedKeysOptimized(NioEventLoop.java:468)
    ##  at io.netty.channel.nio.NioEventLoop.processSelectedKeys(NioEventLoop.java:382)
    ##  at io.netty.channel.nio.NioEventLoop.run(NioEventLoop.java:354)
    ##  at io.netty.util.concurrent.SingleThreadEventExecutor$2.run(SingleThreadEventExecutor.java:111)
    ##  at io.netty.util.concurrent.DefaultThreadFactory$DefaultRunnableDecorator.run(DefaultThreadFactory.java:137)
    ##  at java.lang.Thread.run(Unknown Source)
    ## 17/07/11 10:18:41 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
    ## 17/07/11 10:18:42 INFO SecurityManager: Changing view acls to: lzq
    ## 17/07/11 10:18:42 INFO SecurityManager: Changing modify acls to: lzq
    ## 17/07/11 10:18:42 INFO SecurityManager: Changing view acls groups to: 
    ## 17/07/11 10:18:42 INFO SecurityManager: Changing modify acls groups to: 
    ## 17/07/11 10:18:42 INFO SecurityManager: SecurityManager: authentication disabled; ui acls disabled; users  with view permissions: Set(lzq); groups with view permissions: Set(); users  with modify permissions: Set(lzq); groups with modify permissions: Set()
    ## 17/07/11 10:18:42 INFO Utils: Successfully started service 'sparkDriver' on port 63762.
    ## 17/07/11 10:18:42 INFO SparkEnv: Registering MapOutputTracker
    ## 17/07/11 10:18:42 INFO SparkEnv: Registering BlockManagerMaster
    ## 17/07/11 10:18:42 INFO DiskBlockManager: Created local directory at C:\Users\lzq\AppData\Local\Temp\blockmgr-64509ed9-4ce4-4e8c-8ea8-c54280ccd575
    ## 17/07/11 10:18:42 INFO MemoryStore: MemoryStore started with capacity 413.9 MB
    ## 17/07/11 10:18:42 INFO SparkEnv: Registering OutputCommitCoordinator
    ## 17/07/11 10:18:42 INFO Utils: Successfully started service 'SparkUI' on port 4040.
    ## 17/07/11 10:18:42 INFO SparkUI: Bound SparkUI to 127.0.0.1, and started at http://127.0.0.1:4040
    ## 17/07/11 10:18:42 INFO SparkContext: Added JAR file:/D:/R/R-3.4.1/library/sparklyr/java/sparklyr-2.0-2.11.jar at spark://127.0.0.1:63762/jars/sparklyr-2.0-2.11.jar with timestamp 1499782722652
    ## 17/07/11 10:18:42 INFO Executor: Starting executor ID driver on host localhost
    ## 17/07/11 10:18:42 INFO Utils: Successfully started service 'org.apache.spark.network.netty.NettyBlockTransferService' on port 63783.
    ## 17/07/11 10:18:42 INFO NettyBlockTransferService: Server created on 127.0.0.1:63783
    ## 17/07/11 10:18:42 INFO BlockManagerMaster: Registering BlockManager BlockManagerId(driver, 127.0.0.1, 63783)
    ## 17/07/11 10:18:42 INFO BlockManagerMasterEndpoint: Registering block manager 127.0.0.1:63783 with 413.9 MB RAM, BlockManagerId(driver, 127.0.0.1, 63783)
    ## 17/07/11 10:18:42 INFO BlockManagerMaster: Registered BlockManager BlockManagerId(driver, 127.0.0.1, 63783)
    ## 17/07/11 10:18:43 WARN SparkContext: Use an existing SparkContext, some configuration may not take effect.
    ## 17/07/11 10:18:43 WARN SparkContext: Use an existing SparkContext, some configuration may not take effect.
    ## 17/07/11 10:18:43 INFO HiveSharedState: Warehouse path is 'C:UserslzqAppDataLocal
    ## studiosparkCachespark-2.0.2-bin-hadoop2.7    mphive'.

``` r
# look into the contents of the hive-site.xml Spark config 
cat(paste(readLines(file.path(spark_home_dir(), "conf", "hive-site.xml")), collapse = "\n"))
```

    ## <configuration>
    ##   <property>
    ##     <name>javax.jdo.option.ConnectionURL</name>
    ##     <value>jdbc:derby:memory:databaseName=metastore_db;create=true</value>
    ##   </property>
    ##   <property>
    ##     <name>javax.jdo.option.ConnectionDriverName</name>
    ##     <value>org.apache.derby.jdbc.EmbeddedDriver</value>
    ##   </property>
    ##   <property>
    ##     <name>hive.exec.scratchdir</name>
    ##     <value>C:\Users\lzq\AppData\Local\rstudio\spark\Cache\spark-2.0.2-bin-hadoop2.7\tmp\hive</value>
    ##   </property>
    ##   <property>
    ##     <name>hive.exec.local.scratchdir</name>
    ##     <value>C:\Users\lzq\AppData\Local\rstudio\spark\Cache\spark-2.0.2-bin-hadoop2.7\tmp\hive</value>
    ##   </property>
    ##   <property>
    ##     <name>hive.metastore.warehouse.dir</name>
    ##     <value>C:\Users\lzq\AppData\Local\rstudio\spark\Cache\spark-2.0.2-bin-hadoop2.7\tmp\hive</value>
    ##   </property>
    ## </configuration>

#### Try this.

``` r
Sys.setenv(SPARK_HOME="C:/Users/lzq/AppData/Local/rstudio/spark/Cache/spark-2.0.2-bin-hadoop2.7")
spark_home_dir()
```

    ## [1] "C:\\Users\\lzq\\AppData\\Local\\rstudio\\spark\\Cache/spark-2.0.2-bin-hadoop2.7"

``` r
config <- spark_config()
hadoopBin <- paste0("file:", normalizePath(file.path(spark_home_dir(), "tmp", "hadoop", "bin")))
#hadoopBin <- paste0("file://", normalizePath(file.path(spark_home_dir(), "tmp", "hadoop", "bin")))
#hadoopBin <- paste0("\"", normalizePath(file.path(spark_home_dir(), "tmp", "hadoop", "bin")), "\"")
#hadoopBin <- paste0("'", normalizePath(file.path(spark_home_dir(), "tmp", "hadoop", "bin")), "'")
config[["spark.sql.warehouse.dir"]] <- if (.Platform$OS.type == "windows") hadoopBin else NULL

hiveBin<- paste0("file://", normalizePath(file.path(spark_home_dir(), "tmp", "hadoop", "bin")))
config[["hive.metastore.warehouse.dir"]] <- if (.Platform$OS.type == "windows") hiveBin else NULL

sc <- spark_connect(master = "local",config = config)
```

    ## Re-using existing Spark connection to local

``` r
sc <- spark_connect(master = "local", config = list(spark.sql.warehouse.dir = "c:\\Users\\lzq\\AppData\\Local\\rstudio\\spark\\Cache/spark-2.0.2-bin-hadoop2.7/tmp/hadoop/bin"))

#iris_tbl <- copy_to(sc, iris, overwrite = TRUE)
```

#### Not working. Try this too

``` r
config <- spark_config()
config[["spark.sql.hive.thriftServer.singleSession"]] <- "true"
sc <- spark_connect(master = "local", config = config)
```

    ## Re-using existing Spark connection to local

``` r
#iris_tbl <- copy_to(sc, iris, overwrite = TRUE)
```

``` r
spark_log(sc)
```

    ##  at io.netty.util.concurrent.DefaultThreadFactory$DefaultRunnableDecorator.run(DefaultThreadFactory.java:137)
    ##  at java.lang.Thread.run(Unknown Source)
    ## 17/07/11 10:18:41 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
    ## 17/07/11 10:18:42 INFO SecurityManager: Changing view acls to: lzq
    ## 17/07/11 10:18:42 INFO SecurityManager: Changing modify acls to: lzq
    ## 17/07/11 10:18:42 INFO SecurityManager: Changing view acls groups to: 
    ## 17/07/11 10:18:42 INFO SecurityManager: Changing modify acls groups to: 
    ## 17/07/11 10:18:42 INFO SecurityManager: SecurityManager: authentication disabled; ui acls disabled; users  with view permissions: Set(lzq); groups with view permissions: Set(); users  with modify permissions: Set(lzq); groups with modify permissions: Set()
    ## 17/07/11 10:18:42 INFO Utils: Successfully started service 'sparkDriver' on port 63762.
    ## 17/07/11 10:18:42 INFO SparkEnv: Registering MapOutputTracker
    ## 17/07/11 10:18:42 INFO SparkEnv: Registering BlockManagerMaster
    ## 17/07/11 10:18:42 INFO DiskBlockManager: Created local directory at C:\Users\lzq\AppData\Local\Temp\blockmgr-64509ed9-4ce4-4e8c-8ea8-c54280ccd575
    ## 17/07/11 10:18:42 INFO MemoryStore: MemoryStore started with capacity 413.9 MB
    ## 17/07/11 10:18:42 INFO SparkEnv: Registering OutputCommitCoordinator
    ## 17/07/11 10:18:42 INFO Utils: Successfully started service 'SparkUI' on port 4040.
    ## 17/07/11 10:18:42 INFO SparkUI: Bound SparkUI to 127.0.0.1, and started at http://127.0.0.1:4040
    ## 17/07/11 10:18:42 INFO SparkContext: Added JAR file:/D:/R/R-3.4.1/library/sparklyr/java/sparklyr-2.0-2.11.jar at spark://127.0.0.1:63762/jars/sparklyr-2.0-2.11.jar with timestamp 1499782722652
    ## 17/07/11 10:18:42 INFO Executor: Starting executor ID driver on host localhost
    ## 17/07/11 10:18:42 INFO Utils: Successfully started service 'org.apache.spark.network.netty.NettyBlockTransferService' on port 63783.
    ## 17/07/11 10:18:42 INFO NettyBlockTransferService: Server created on 127.0.0.1:63783
    ## 17/07/11 10:18:42 INFO BlockManagerMaster: Registering BlockManager BlockManagerId(driver, 127.0.0.1, 63783)
    ## 17/07/11 10:18:42 INFO BlockManagerMasterEndpoint: Registering block manager 127.0.0.1:63783 with 413.9 MB RAM, BlockManagerId(driver, 127.0.0.1, 63783)
    ## 17/07/11 10:18:42 INFO BlockManagerMaster: Registered BlockManager BlockManagerId(driver, 127.0.0.1, 63783)
    ## 17/07/11 10:18:43 WARN SparkContext: Use an existing SparkContext, some configuration may not take effect.
    ## 17/07/11 10:18:43 WARN SparkContext: Use an existing SparkContext, some configuration may not take effect.
    ## 17/07/11 10:18:43 INFO HiveSharedState: Warehouse path is 'C:UserslzqAppDataLocal
    ## studiosparkCachespark-2.0.2-bin-hadoop2.7    mphive'.
    ## 17/07/11 10:18:48 INFO SparkContext: Running Spark version 2.0.2
    ## 17/07/11 10:18:48 ERROR Shell: Failed to locate the winutils binary in the hadoop binary path
    ## java.io.IOException: Could not locate executable C:\Users\lzq\AppData\Local\rstudio\spark\Cache\spark-2.0.2-bin-hadoop2.7\tmp\hadoop\bin\bin\winutils.exe in the Hadoop binaries.
    ##  at org.apache.hadoop.util.Shell.getQualifiedBinPath(Shell.java:379)
    ##  at org.apache.hadoop.util.Shell.getWinUtilsPath(Shell.java:394)
    ##  at org.apache.hadoop.util.Shell.<clinit>(Shell.java:387)
    ##  at org.apache.hadoop.util.StringUtils.<clinit>(StringUtils.java:80)
    ##  at org.apache.hadoop.security.SecurityUtil.getAuthenticationMethod(SecurityUtil.java:611)
    ##  at org.apache.hadoop.security.UserGroupInformation.initialize(UserGroupInformation.java:273)
    ##  at org.apache.hadoop.security.UserGroupInformation.ensureInitialized(UserGroupInformation.java:261)
    ##  at org.apache.hadoop.security.UserGroupInformation.loginUserFromSubject(UserGroupInformation.java:791)
    ##  at org.apache.hadoop.security.UserGroupInformation.getLoginUser(UserGroupInformation.java:761)
    ##  at org.apache.hadoop.security.UserGroupInformation.getCurrentUser(UserGroupInformation.java:634)
    ##  at org.apache.spark.util.Utils$$anonfun$getCurrentUserName$1.apply(Utils.scala:2345)
    ##  at org.apache.spark.util.Utils$$anonfun$getCurrentUserName$1.apply(Utils.scala:2345)
    ##  at scala.Option.getOrElse(Option.scala:121)
    ##  at org.apache.spark.util.Utils$.getCurrentUserName(Utils.scala:2345)
    ##  at org.apache.spark.SparkContext.<init>(SparkContext.scala:294)
    ##  at org.apache.spark.SparkContext$.getOrCreate(SparkContext.scala:2258)
    ##  at org.apache.spark.SparkContext.getOrCreate(SparkContext.scala)
    ##  at sun.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
    ##  at sun.reflect.NativeMethodAccessorImpl.invoke(Unknown Source)
    ##  at sun.reflect.DelegatingMethodAccessorImpl.invoke(Unknown Source)
    ##  at java.lang.reflect.Method.invoke(Unknown Source)
    ##  at sparklyr.Invoke$.invoke(invoke.scala:94)
    ##  at sparklyr.StreamHandler$.handleMethodCall(stream.scala:89)
    ##  at sparklyr.StreamHandler$.read(stream.scala:55)
    ##  at sparklyr.BackendHandler.channelRead0(handler.scala:49)
    ##  at sparklyr.BackendHandler.channelRead0(handler.scala:14)
    ##  at io.netty.channel.SimpleChannelInboundHandler.channelRead(SimpleChannelInboundHandler.java:105)
    ##  at io.netty.channel.AbstractChannelHandlerContext.invokeChannelRead(AbstractChannelHandlerContext.java:308)
    ##  at io.netty.channel.AbstractChannelHandlerContext.fireChannelRead(AbstractChannelHandlerContext.java:294)
    ##  at io.netty.handler.codec.MessageToMessageDecoder.channelRead(MessageToMessageDecoder.java:103)
    ##  at io.netty.channel.AbstractChannelHandlerContext.invokeChannelRead(AbstractChannelHandlerContext.java:308)
    ##  at io.netty.channel.AbstractChannelHandlerContext.fireChannelRead(AbstractChannelHandlerContext.java:294)
    ##  at io.netty.handler.codec.ByteToMessageDecoder.channelRead(ByteToMessageDecoder.java:244)
    ##  at io.netty.channel.AbstractChannelHandlerContext.invokeChannelRead(AbstractChannelHandlerContext.java:308)
    ##  at io.netty.channel.AbstractChannelHandlerContext.fireChannelRead(AbstractChannelHandlerContext.java:294)
    ##  at io.netty.channel.DefaultChannelPipeline.fireChannelRead(DefaultChannelPipeline.java:846)
    ##  at io.netty.channel.nio.AbstractNioByteChannel$NioByteUnsafe.read(AbstractNioByteChannel.java:131)
    ##  at io.netty.channel.nio.NioEventLoop.processSelectedKey(NioEventLoop.java:511)
    ##  at io.netty.channel.nio.NioEventLoop.processSelectedKeysOptimized(NioEventLoop.java:468)
    ##  at io.netty.channel.nio.NioEventLoop.processSelectedKeys(NioEventLoop.java:382)
    ##  at io.netty.channel.nio.NioEventLoop.run(NioEventLoop.java:354)
    ##  at io.netty.util.concurrent.SingleThreadEventExecutor$2.run(SingleThreadEventExecutor.java:111)
    ##  at io.netty.util.concurrent.DefaultThreadFactory$DefaultRunnableDecorator.run(DefaultThreadFactory.java:137)
    ##  at java.lang.Thread.run(Unknown Source)
    ## 17/07/11 10:18:48 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
    ## 17/07/11 10:18:49 INFO SecurityManager: Changing view acls to: lzq
    ## 17/07/11 10:18:49 INFO SecurityManager: Changing modify acls to: lzq
    ## 17/07/11 10:18:49 INFO SecurityManager: Changing view acls groups to: 
    ## 17/07/11 10:18:49 INFO SecurityManager: Changing modify acls groups to: 
    ## 17/07/11 10:18:49 INFO SecurityManager: SecurityManager: authentication disabled; ui acls disabled; users  with view permissions: Set(lzq); groups with view permissions: Set(); users  with modify permissions: Set(lzq); groups with modify permissions: Set()
    ## 17/07/11 10:18:49 INFO Utils: Successfully started service 'sparkDriver' on port 63811.
    ## 17/07/11 10:18:49 INFO SparkEnv: Registering MapOutputTracker
    ## 17/07/11 10:18:49 INFO SparkEnv: Registering BlockManagerMaster
    ## 17/07/11 10:18:49 INFO DiskBlockManager: Created local directory at C:\Users\lzq\AppData\Local\Temp\blockmgr-f0af8e7e-bd02-41c4-8bb7-96e86a6bee94
    ## 17/07/11 10:18:49 INFO MemoryStore: MemoryStore started with capacity 413.9 MB
    ## 17/07/11 10:18:49 INFO SparkEnv: Registering OutputCommitCoordinator
    ## 17/07/11 10:18:49 WARN Utils: Service 'SparkUI' could not bind on port 4040. Attempting port 4041.
    ## 17/07/11 10:18:49 INFO Utils: Successfully started service 'SparkUI' on port 4041.
    ## 17/07/11 10:18:49 INFO SparkUI: Bound SparkUI to 0.0.0.0, and started at http://192.168.1.151:4041
    ## 17/07/11 10:18:49 INFO SparkContext: Added JAR file:/D:/R/R-3.4.1/library/sparklyr/java/sparklyr-2.0-2.11.jar at spark://192.168.1.151:63811/jars/sparklyr-2.0-2.11.jar with timestamp 1499782729670
    ## 17/07/11 10:18:49 INFO Executor: Starting executor ID driver on host localhost
    ## 17/07/11 10:18:49 INFO Utils: Successfully started service 'org.apache.spark.network.netty.NettyBlockTransferService' on port 63820.
    ## 17/07/11 10:18:49 INFO NettyBlockTransferService: Server created on 192.168.1.151:63820
    ## 17/07/11 10:18:49 INFO BlockManagerMaster: Registering BlockManager BlockManagerId(driver, 192.168.1.151, 63820)
    ## 17/07/11 10:18:49 INFO BlockManagerMasterEndpoint: Registering block manager 192.168.1.151:63820 with 413.9 MB RAM, BlockManagerId(driver, 192.168.1.151, 63820)
    ## 17/07/11 10:18:49 INFO BlockManagerMaster: Registered BlockManager BlockManagerId(driver, 192.168.1.151, 63820)
    ## 17/07/11 10:18:50 WARN SparkContext: Use an existing SparkContext, some configuration may not take effect.
    ## 17/07/11 10:18:50 WARN SparkContext: Use an existing SparkContext, some configuration may not take effect.
    ## 17/07/11 10:18:50 INFO HiveSharedState: Warehouse path is 'C:UserslzqAppDataLocal
    ## studiosparkCachespark-2.0.2-bin-hadoop2.7    mphive'.

I put a lot of efforts to make sparklyr work in Windows 10 but failed to wirte data into spark. I posted the issue in RStudion/sparklyr for help but did not get response yet. So I could not finish project 5 in time. I will try SparkR. Hopefully I can figure out how to use Spark in R and apply it in the final project.
