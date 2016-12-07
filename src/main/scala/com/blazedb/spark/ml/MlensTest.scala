package org.apache.spark.sql
//package com.blazedb.spark.mllib

import org.apache.spark.ml.recommendation._
import org.apache.spark.ml._
import org.apache.spark._
import org.apache.spark.ml.recommendation.ALS.Rating
import org.apache.spark.rdd.RDD
//import org.apache.spark.sql.SQLContext

object MlensTest {

//  type Rating = case class Rating(user: Int, item: Int, rating: Double)

  type ID = Int
  type Rat = Rating[ID]
  type UIRdd =  RDD[(ID, Array[Float])]
  def downloadMlens(workDir: String) = {
    val smallDsUrl = "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    val blob = scala.io.Source.fromInputStream(
      new java.net.URL(smallDsUrl).openConnection.getInputStream,
      "ISO-8859-1").getLines().mkString("\n")
    blob
  }

  def loadMlens(sc: SparkContext, workDir: String) = {
    val dspath = "data/movielens"
    val smallDs = s"$workDir/$dspath/ml-latest-small.zip"
    val outDat = sc.textFile(smallDs)
    outDat.map { o =>
      val toks = o.split(",")
      Rating[Int](o(0).toInt, o(1).toInt, o(2).toFloat)
    }
  }

  def buildModel(dat: RDD[Rat]) = {

  }

  def main(args: Array[String]) = {
    val master = args(0)

    val Factors = 100
    val Iters = 10
    val Lambda = 0.01
    val UserBlocks = 20
    val ItemBlocks = 20
    val sparkConf = new SparkConf().setMaster(master).setAppName("ALSTest")
    val sc = new SparkContext(sparkConf)
    val sqlc = new SQLContext(sc)
    val parsed = loadMlens(sc, "/tmp/mlens")
//    parsed.toDF()
    val df = sqlc.createDataFrame(parsed)
    df.cache
//    val model = ALS.train(parsed,Factors, Iters, Lambda, Blocks)
    val als = new ALS()
      .setRank(Factors)
      .setRegParam(Lambda)
      .setImplicitPrefs(false)
      .setNumUserBlocks(UserBlocks)
      .setNumItemBlocks(ItemBlocks)
      .setSeed(0)

//    val (userRdd, itemRdd) = als.fit(df)
    val model = als.fit(df)
//    val predicted = model.
  }
}

