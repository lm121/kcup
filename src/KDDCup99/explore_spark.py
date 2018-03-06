#!/usr/bin/env python


import KDDCup99_common as kup
import sys
import os


from collections import OrderedDict
from numpy import array
from math import sqrt


try:
    from pyspark import SparkContext, SparkConf
    from pyspark.sql import SparkSession
    from pyspark.sql.types import *   
    from pyspark.mllib.clustering import KMeans
    from pyspark.mllib.feature import StandardScaler
    from pyspark.mllib.evaluation import MulticlassMetrics
    from pyspark.mllib.linalg import Vectors
    from pyspark.mllib.regression import LabeledPoint
    from pyspark.mllib.classification import LogisticRegressionWithLBFGS
    from pyspark.ml.feature import OneHotEncoder, StringIndexer
    from pyspark.ml.feature import VectorIndexer
    print ("Successfully imported Spark Modules")
except ImportError as e:
    print ("Can not import Spark Modules", e)
    sys.exit(1)

def testVectorIndexer(spark,data):

    indexer = VectorIndexer(inputCol="features", outputCol="indexed", maxCategories=10)
    indexerModel = indexer.fit(data)

    categoricalFeatures = indexerModel.categoryMaps
    print("Chose %d categorical features: %s" %
          (len(categoricalFeatures), ", ".join(str(k) for k in categoricalFeatures.keys())))

    # Create new column "indexed" with categorical values transformed to indices
    indexedData = indexerModel.transform(data)
    indexedData.show()


def testOneHotEncoder(spark):
    df = spark.createDataFrame([
        (0, "a"),
        (1, "b"),
        (2, "c"),
        (3, "a"),
        (4, "a"),
        (5, "c")
    ], ["id", "category"])
    testOneHotEncoderDF(spark,df,"category","categoryIndex")

def testOneHotEncoderDF(spark,df,incol,outcol):
    stringIndexer = StringIndexer(inputCol=incol, outputCol=outcol)
    model = stringIndexer.fit(df)
    indexed = model.transform(df)
    indexed.printSchema()
    indexed.show(3)


if __name__ == "__main__":
    if (len(sys.argv) < 3):
        print "Usage: spark-submit --driver-memory 2g " + \
          "KDDCup99_logreg.py kddcup.data.file golden.file"
        sys.exit(1)

    # set up environment

    data_file = sys.argv[1]
    golden_file=sys.argv[2]


    spark = SparkSession \
        .builder \
        .appName("KDDCup99_logreg") \
        .config("spark.executor.memory", "2g") \
        .getOrCreate()

    sc = spark.sparkContext

    
    log4jLogger = spark._jvm.org.apache.log4j 
    log4jLogger.LogManager.getLogger("org").setLevel(log4jLogger.Level.WARN)
    log4jLogger.LogManager.getLogger("akka").setLevel(log4jLogger.Level.OFF)

    print("")
    
    raw_data_text = sc.textFile(data_file)
    parsed_data = raw_data_text.map(kup.parse_interaction)
    parsed_labels = parsed_data.map(lambda x: str(x[0]))
    print("parsed_labels: "+str(type(parsed_labels)))
    df=spark.createDataFrame(parsed_labels,StringType()).toDF("label")
    df.printSchema()
    df.show(3)
    testOneHotEncoderDF(spark,df,"label","labelc")
    if 1 ==0:
        print "Loading RAW data..."
        
        raw_data=spark.read.csv(data_file)
    
        print "Parsing dataset..."
        fields = [StructField(field_name, StringType(), True) for field_name in kup.col_names]
        schema = StructType(fields)
        df = spark.createDataFrame(raw_data.rdd, schema)


        testOneHotEncoderDF(spark,df,"label","labelc")
      
    print "DONE!"













