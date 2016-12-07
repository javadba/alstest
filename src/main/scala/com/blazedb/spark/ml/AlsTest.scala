package com.blazedb.spark.ml

import java.io.File
import java.util.Random

import breeze.linalg.rank
import org.apache.spark.ml.recommendation._
import org.apache.spark.ml._
import org.apache.spark._
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.recommendation.ALS.Rating
import org.apache.spark.ml.util.DefaultReadWriteTest
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.storage.StorageLevel
//import org.apache.spark.sql.SQLContext
import org.scalatest.FunSuite
import com.github.fommil.netlib.BLAS.{getInstance => blas}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

class ExplicitAlsTest
  extends FunSuite with MLlibTestSparkContext with DefaultReadWriteTest with Logging {

}

//  type Rating = case class Rating(user: Int, item: Int, rating: Double)

object AlsTest {

  case class AlsParams(users: Int, items: Int, userBlocks: Int, itemBlocks: Int, factors: Int, iters: Int,
    regLambda: Double = 1e-4, rmse: Double = 2e-3, noiseStdev: Double = 1e-2) {
    override def toString() = {
      s"ALS: users=$users items=$items ublocks=$userBlocks iblocks=$itemBlocks factors=$factors" +
        s" iters=$iters lambda=$regLambda rmse=$rmse noise=$noiseStdev"
    }
  }

  def blastorama(sc: SparkContext, sqlc: SQLContext, alsp: AlsParams) = {
    val start = System.currentTimeMillis
    val (training, test) =
      genImplicitTestData(sc, alsp.users, alsp.items, alsp.factors, alsp.noiseStdev)
    println(s"Generated dataset in ${(System.currentTimeMillis - start)}ms")
    val start2 = System.currentTimeMillis
    testALS(sqlc, training, test, maxIter = alsp.iters, rank = alsp.factors, regParam = alsp.regLambda, targetRMSE = alsp.rmse,
      numItemBlocks = alsp.itemBlocks, numUserBlocks = alsp.userBlocks)
    println(s"ALS Test ran in ${((System.currentTimeMillis - start2) / 100).toInt / 10}secs")

    //    val (training, test) =
    //      genImplicitTestData(numUsers = 20, numItems = 40, rank = 2, noiseStd = 0.01)
    //    testALS(training, test, maxIter = 4, rank = 2, regParam = 0.01, implicitPrefs = true,
    //      targetRMSE = 0.3)
  }

  def logError(msg: String) = logInfo(s"ERROR: $msg")

  def logInfo(msg: String) = {
    val d = new java.util.Date().toString.substring(4, 19)
    println(s"[$d] $msg")
  }

  /**
    * Generates an explicit feedback dataset for testing ALS.
    *
    * @param numUsers number of users
    * @param numItems number of items
    * @param rank     rank
    * @param noiseStd the standard deviation of additive Gaussian noise on training data
    * @param seed     random seed
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
    *
    * @param numUsers number of users
    * @param numItems number of items
    * @param rank     rank
    * @param noiseStd the standard deviation of additive Gaussian noise on training data
    * @param seed     random seed
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
    *
    * @param size   number of users/items
    * @param rank   number of features
    * @param random random number generator
    * @param a      min value of the support (default: -1)
    * @param b      max value of the support (default: 1)
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

  /**
    * Test ALS using the given training/test splits and parameters.
    *
    * @param training      training dataset
    * @param test          test dataset
    * @param rank          rank of the matrix factorization
    * @param maxIter       max number of iterations
    * @param regParam      regularization constant
    * @param implicitPrefs whether to use implicit preference
    * @param numUserBlocks number of user blocks
    * @param numItemBlocks number of item blocks
    * @param targetRMSE    target test RMSE
    */
  def testALS(
    sqlc: SQLContext,
    training: RDD[Rating[Int]],
    test: RDD[Rating[Int]],
    rank: Int,
    maxIter: Int,
    regParam: Double,
    implicitPrefs: Boolean = false,
    numUserBlocks: Int = 2,
    numItemBlocks: Int = 3,
    targetRMSE: Double = 0.05): Unit = {
    import sqlc.implicits._
    training.persist(StorageLevel.DISK_ONLY)
    test.persist(StorageLevel.DISK_ONLY)
    val als = new ALS()
      .setRank(rank)
      .setRegParam(regParam)
      .setImplicitPrefs(implicitPrefs)
      .setNumUserBlocks(numUserBlocks)
      .setNumItemBlocks(numItemBlocks)
        .setCheckpointInterval(1)
      .setSeed(0)
    val alpha = als.getAlpha
    val model = als.fit(training.toDF())
    val predictions = model.transform(test.toDF())
      .select("rating", "prediction")
      .map { case Row(rating: Float, prediction: Float) =>
        (rating.toDouble, prediction.toDouble)
      }
    val rmse =
      if (implicitPrefs) {
        // TODO: Use a better (rank-based?) evaluation metric for implicit feedback.
        // We limit the ratings and the predictions to interval [0, 1] and compute the weighted RMSE
        // with the confidence scores as weights.
        val (totalWeight, weightedSumSq) = predictions.map { case (rating, prediction) =>
          val confidence = 1.0 + alpha * math.abs(rating)
          val rating01 = math.max(math.min(rating, 1.0), 0.0)
          val prediction01 = math.max(math.min(prediction, 1.0), 0.0)
          val err = prediction01 - rating01
          (confidence, confidence * err * err)
        }.reduce { case ((c0, e0), (c1, e1)) =>
          (c0 + c1, e0 + e1)
        }
        math.sqrt(weightedSumSq / totalWeight)
      } else {
        val mse = predictions.map { case (rating, prediction) =>
          val err = rating - prediction
          err * err
        }.mean()
        math.sqrt(mse)
      }
    logInfo(s"Test RMSE is $rmse.")
    if (rmse < targetRMSE) {
      logError(s"rmse=$rmse whereas we kinda figured $targetRMSE")
    }

    // copied model must have the same parent.
    checkCopy(model)
  }

  def checkCopy(model: Model[_]): Unit = {
    val copied = model.copy(ParamMap.empty)
      .asInstanceOf[Model[_]]
    assert(copied.parent.uid == model.parent.uid)
    assert(copied.parent == model.parent)
  }


  def main(args: Array[String]) = {
    if (args.length < 6) {
      println("Usage: AlsTest master users items userBlocks itemBlocks factors iters L2Lambda rmse noise-std")
      System.exit(1)
    }
    var i = 0
    val master = args(i)
    i += 1
    val users = args(i).toInt
    i += 1
    val items = args(i).toInt
    i += 1
    val userBlocks = args(i).toInt
    i += 1
    val itemBlocks = args(i).toInt
    i += 1
    val factors = args(i).toInt
    i += 1
    val iters = args(i).toInt
    i += 1
    val regLambda = args(i).toDouble
    i += 1
    val rmse = args(i).toDouble
    i += 1
    val noiseStdev = if (args.length >= 10) args(i).toDouble else 0.01
    i += 1
    val alsp = AlsParams(users, items, userBlocks, itemBlocks, factors, iters, regLambda, rmse, noiseStdev)
    println(s"Running ALSTest with $alsp")
    val sparkConf = new SparkConf().setMaster(master).setAppName("ALSTest")
    val sc = new SparkContext(sparkConf)
    // val host = java.net.InetAddress.getLocalHost.getHostName
    //    val tempDir = File.createTempFile(s"hdfs://$host:8021/tmp/alsTest","tmp")
    val tempDir = s"/data/check/alsTest"
    sc.setCheckpointDir(tempDir)
    val sqlc = new SQLContext(sc)
    val res = blastorama(sc, sqlc, alsp)

  }
}

