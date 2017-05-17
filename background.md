### Stephen Background sound bites

- 2014-present: **Spark/Scala for data engineering and/or data science/machine learning**
  - Scaling/integration of Recommendation Systems Data Science pipelines 
     - topK movies/shows by category
     - recommendations per user based on cluster affinity and per user taste vector
         
- 2011-2014: **Hadoop/Hive/Map-Reduce/ETL**
- 2008-2011: <b>Server side java systems</b>: messaging, caching
- Until 1996:  Mix of <b>C/C++ Assembler</b> with some VLSI and sql thrown in 
  - IBM Research award for automated VLSI chip testing suites 1991
  - Specialty was multithreaded network programming
  - Part of IBM Research team for first known internet enabled multi stream MPEG video client 1994-1995
- 1996-2007: <b>Server side Java and Oracle developer/dba</b>
  - java Multithreaded video server ibm research 1996
  - java/corba and oracle dev work 1999
  - Developed/owned data dictionary, Oracle instance, java ORM layer, and reporting for HP eCommerce site 1999-2001
  - Designed and implemented Data Change Capture system for HP worldwide product catalogs supporting 23 upstream systems 2006-2007
  - Primary developer and maintainer full stack mobile application (linux, oracle -> mysql, j2ee -> pure java, voice xml/web tech) 2002-2011

Projects


- Spark Development to support distributed GPU computing:  Point to Point (direct RPC) RDD and Locality Sensitive RDD. Also TensorFlow integration and support for Native DMA Channels  (mainly side work may 2016 to april 2017)

  This repo is 100% my code, and mostly on the side except for small piece funded by OpenChai a year ago.

This piece is a custom RPC mechanism used for doing gradient descent updates for GPU driven algorithms without requiring reloads via spark .  The driver communicates with the spark workers via side-band TCP channels to receive their weights and send back the best weights seen so far (a configurable approach as opposed to averaging the weights)

[P2p RDD](https://github.com/OpenChaiSpark/OCspark/tree/master/p2prdd/src/main/scala/org/openchai/spark/rdd) 

The solver interface is here:

[Solver IF](https://github.com/OpenChaiSpark/OCspark/blob/master/tcpclient/src/main/scala/org/openchai/tcp/rpc/SolverIf.scala)


- Spark Performance Testing  (july to sept 2015)  This one was partially a small project funded by GridGain  and I made some updates for Spark As a Service development at eBay.   At eBay a derivative was used for loading 90%+ cpu and memory across the 20 machines (16cpus  and 128GB RAM apiece).   I worked on all the parameters for memory, cpu across yarn, mesos, and standalone. Yarn was the trickiest to configure : we had to lie about the memory available to get above 90% utilization.

[Spark Perf](https://github.com/javadba/sparkperf)   


One of the tests: core RDD tests

[Core RDD Test](https://github.com/javadba/sparkperf/blob/master/src/main/scala/com/blazedb/sparkperf/CoreRDDTest.scala)

[Cpu load Test](https://github.com/javadba/sparkperf/blob/master/src/main/scala/com/blazedb/sparkperf/CpuTest.scala )


[Data Generator to stress CPU and Memory](https://github.com/javadba/sparkperf/blob/master/src/main/scala/com/blazedb/sparkperf/DataGenerator.scala) 


- Spark MLLib contribution for Spectral Clustering - which was changed to Power Iteration Clustering. This is mainly of historical interest since the code has not been well maintained since its merging into Spark 1.3.0 in january 2015

  - Key pieces of my original Spectral Clustering code including using Schur's complement on iterated principal eigenvectors:

[Spectral Clustering](https://github.com/Huawei-Spark/spark/blob/43ab10be1c634f88d08f666df71ff15427e8a3d2/mllib/src/main/scala/org/apache/spark/mllib/clustering/RDDLinalg.scala)

[Power Iteration Clustering](https://github.com/Huawei-Spark/spark/blob/43ab10be1c634f88d08f666df71ff15427e8a3d2/mllib/src/main/scala/org/apache/spark/mllib/clustering/PIClustering.scala)

[Power Iteration Linear Algebra](https://github.com/Huawei-Spark/spark/blob/43ab10be1c634f88d08f666df71ff15427e8a3d2/mllib/src/main/scala/org/apache/spark/mllib/clustering/PICLinalg.scala)


- Later Databricks decided they wanted to use Graphx so it was completely rewritten. Here is that version:

[Power Iteration Clustering via Graphx](https://github.com/apache/spark/commit/377431a578f621b599b538f069adca6accaf7a9)


     A blog prepared for use jointly with DataBricks / Xiangrui Meng. But they opted for much shorter version.

[Proposed Join Blog with Databricks](https://drive.google.com/file/d/0B4k_P472xo72b1RrSkNoMW9TN1VQeU5lcVdrdDJSdWJtVXBj/view?usp=sharing)

  
  

   



