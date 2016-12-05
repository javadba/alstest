package com.blazedb.spark.ml

import java.io.File
import java.util.Random

import breeze.linalg.rank
import org.apache.spark.ml.recommendation._
import org.apache.spark.ml._
import org.apache.spark._
import org.apache.spark.ml.recommendation.ALS.Rating
import org.apache.spark.ml.util.DefaultReadWriteTest
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.rdd.RDD
//import org.apache.spark.sql.SQLContext
import org.scalatest.FunSuite
import com.github.fommil.netlib.BLAS.{getInstance => blas}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

class ExplicitAlsTest
  extends FunSuite with MLlibTestSparkContext with DefaultReadWriteTest with Logging {

}

//  type Rating = case class Rating(user: Int, item: Int, rating: Double)

object  ExplicitAlsTest {

  def blastorama(sc: SparkContext) = {
    val (ratings, _) = genImplicitTestData(sc, numUsers = 1000, numItems = 800, rank = 20, noiseStd = 0.01)

    val longRatings = ratings.map(r => Rating(r.user.toLong, r.item.toLong, r.rating))
    val (longUserFactors, _) = ALS.train(longRatings, rank = 200, maxIter = 4, seed = 0)
    assert(longUserFactors.first()._1.getClass == classOf[Long], s"Incorrect userfactors class (!=Long)")

    val strRatings = ratings.map(r => Rating(r.user.toString, r.item.toString, r.rating))
    val (strUserFactors, _) = ALS.train(strRatings, rank = 2, maxIter = 4, seed = 0)
    assert(strUserFactors.first()._1.getClass == classOf[String], s"Incorrect userfactors class (!=String)")
  }

  def logInfo(msg: String) = {
    val d = new java.util.Date().toString.substring(4, 19)
    println(s"[$d] $msg")
  }

  /**
   * Generates an explicit feedback dataset for testing ALS.
    *
    * @param numUsers number of users
   * @param numItems number of items
   * @param rank rank
   * @param noiseStd the standard deviation of additive Gaussian noise on training data
   * @param seed random seed
   * @return (training, test)
   */
  def genExplicitTestData(
    sc: SparkContext,
  numUsers: Int,
      numItems: Int,
      rank: Int,
      noiseStd: Double = 0.0,
      seed: Long = 11L): (RDD[Rating[Int]], RDD[Rating[Int]]) = {
    val trainingFraction = 0.6
    val testFraction = 0.3
    val totalFraction = trainingFraction + testFraction
    val random = new Random(seed)
    val userFactors = genFactors(numUsers, rank, random)
    val itemFactors = genFactors(numItems, rank, random)
    val training = ArrayBuffer.empty[Rating[Int]]
    val test = ArrayBuffer.empty[Rating[Int]]
    for ((userId, userFactor) <- userFactors; (itemId, itemFactor) <- itemFactors) {
      val x = random.nextDouble()
      if (x < totalFraction) {
        val rating = blas.sdot(rank, userFactor, 1, itemFactor, 1)
        if (x < trainingFraction) {
          val noise = noiseStd * random.nextGaussian()
          training += Rating(userId, itemId, rating + noise.toFloat)
        } else {
          test += Rating(userId, itemId, rating)
        }
      }
    }
    logInfo(s"Generated an explicit feedback dataset with ${training.size} ratings for training " +
      s"and ${test.size} for test.")
    (sc.parallelize(training, 2), sc.parallelize(test, 2))
  }

  /**
   * Generates an implicit feedback dataset for testing ALS.
   * @param numUsers number of users
   * @param numItems number of items
   * @param rank rank
   * @param noiseStd the standard deviation of additive Gaussian noise on training data
   * @param seed random seed
   * @return (training, test)
   */
  def genImplicitTestData(
    sc: SparkContext,
      numUsers: Int,
      numItems: Int,
      rank: Int,
      noiseStd: Double = 0.0,
      seed: Long = 11L): (RDD[Rating[Int]], RDD[Rating[Int]]) = {
    // The assumption of the implicit feedback model is that unobserved ratings are more likely to
    // be negatives.
    val positiveFraction = 0.8
    val negativeFraction = 1.0 - positiveFraction
    val trainingFraction = 0.6
    val testFraction = 0.3
    val totalFraction = trainingFraction + testFraction
    val random = new Random(seed)
    val userFactors = genFactors(numUsers, rank, random)
    val itemFactors = genFactors(numItems, rank, random)
    val training = ArrayBuffer.empty[Rating[Int]]
    val test = ArrayBuffer.empty[Rating[Int]]
    for ((userId, userFactor) <- userFactors; (itemId, itemFactor) <- itemFactors) {
      val rating = blas.sdot(rank, userFactor, 1, itemFactor, 1)
      val threshold = if (rating > 0) positiveFraction else negativeFraction
      val observed = random.nextDouble() < threshold
      if (observed) {
        val x = random.nextDouble()
        if (x < totalFraction) {
          if (x < trainingFraction) {
            val noise = noiseStd * random.nextGaussian()
            training += Rating(userId, itemId, rating + noise.toFloat)
          } else {
            test += Rating(userId, itemId, rating)
          }
        }
      }
    }
    logInfo(s"Generated an implicit feedback dataset with ${training.size} ratings for training " +
      s"and ${test.size} for test.")
    (sc.parallelize(training, 2), sc.parallelize(test, 2))
  }

  /**
   * Generates random user/item factors, with i.i.d. values drawn from U(a, b).
   * @param size number of users/items
   * @param rank number of features
   * @param random random number generator
   * @param a min value of the support (default: -1)
   * @param b max value of the support (default: 1)
   * @return a sequence of (ID, factors) pairs
   */
  private def genFactors(
      size: Int,
      rank: Int,
      random: Random,
      a: Float = -1.0f,
      b: Float = 1.0f): Seq[(Int, Array[Float])] = {
    require(size > 0 && size < Int.MaxValue / 3)
    require(b > a)
    val ids = mutable.Set.empty[Int]
    while (ids.size < size) {
      ids += random.nextInt()
    }
    val width = b - a
    ids.toSeq.sorted.map(id => (id, Array.fill(rank)(a + random.nextFloat() * width)))
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
    val host = java.net.InetAddress.getLocalHost.getHostName
//    val tempDir = File.createTempFile(s"hdfs://$host:8021/tmp/alsTest","tmp")
    val tempDir = s"/data/check/alsTest"
    sc.setCheckpointDir(tempDir)
//     val sqlc = new SQLContext(sc)
    blastorama(sc)

  }
}

