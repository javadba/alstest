package com.blazedb.spark.ml.util

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row

import org.apache.spark.util.{SizeEstimator => SparkSizeEstimator}

/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
object SizeEstimator {

  def getTotalSize(rdd: RDD[Row], nSamples: Int): Long = {
    // This can be a parameter
    val totalRows = rdd.count()
    var totalSize = 0l
    if (totalRows > nSamples) {
      val sampleRDD = rdd.sample(true, nSamples)
      val sampleRDDSize = getRDDSize(sampleRDD)
      totalSize = sampleRDDSize.*(totalRows)./(nSamples)
    } else {
      // As the RDD is smaller than sample rows count, we can just calculate the total RDD size
      totalSize = getRDDSize(rdd)
    }
    totalSize
  }

//  def getElementSize(rdd: RDD[Row], nSamples: Int): Long = {
//      val sampleRDD = rdd.sample(true, nSamples)
//      val sampleRDDSize = getRDDSize(sampleRDD)
//  }

  def getRDDSize(rdd: RDD[Row]): Long = {
    var rddSize = 0l
    val rows = rdd.collect()
    for (i <- rows.indices) {
      rddSize += SparkSizeEstimator.estimate(rows.apply(i).toSeq.map { value => value.asInstanceOf[AnyRef] })
    }

    rddSize
  }
}
