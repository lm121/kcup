#!/usr/bin/env python


import KDDCup99_common_ml as kup
import sys
import os
import shutil


from collections import OrderedDict
from numpy import array
from math import sqrt

try:
    from pyspark import SparkContext, SparkConf
    from pyspark.sql import SparkSession
    from pyspark.sql.types import *
 
    from pyspark.ml.feature import StandardScaler,StandardScalerModel
    from pyspark.ml.linalg import Vectors
    from pyspark.ml.classification import LogisticRegression,LogisticRegressionModel

    from pyspark.mllib.evaluation import BinaryClassificationMetrics
    from pyspark.mllib.evaluation import RegressionMetrics
    from pyspark.mllib.evaluation import MulticlassMetrics
    
except ImportError as e:
    print ("Can not import Spark Modules", e)
    sys.exit(1)


def printUsageExit(status):
    print "Usage: spark-submit --driver-memory 4g " + \
          "KDDCup99_data_statistics.py [option:stats] [trainData] [testData]"
    sys.exit(status)

if __name__ == "__main__":
    if (len(sys.argv) < 4):
        printUsageExit(1)

    # set up environment
    opt=sys.argv[1]
    if opt == "help" :
        printUsageExit(0)

    data_file = sys.argv[2]
    golden_file=sys.argv[3]


    model_path=kup.saved_model_path+"/logregModel"
    scaler_model_path=kup.saved_model_path+"/scalerModel"

    spark = SparkSession \
        .builder \
        .appName("KDDCup99_data_statistics") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()

    sc = spark.sparkContext

    
    log4jLogger = spark._jvm.org.apache.log4j 
    log4jLogger.LogManager.getLogger("org").setLevel(log4jLogger.Level.WARN)
    log4jLogger.LogManager.getLogger("akka").setLevel(log4jLogger.Level.OFF)

    
    if opt=="stat":

        raw_data = sc.textFile(data_file)
        # count by all different labels and print them decreasingly
        kup.countingLabel(raw_data)
        
      
    if opt =="computeCategoricalData":

        print "Loading RAW data..."
        raw_data = sc.textFile(data_file)

        split_data=raw_data.map(lambda x: x.split(","))
        protocols = split_data.map(lambda x: x[1]).distinct().collect()
        protstr='","'.join(map(str, protocols))

        services = split_data.map(lambda x: x[2]).distinct().collect()
        servicesStr='","'.join(map(str,services))
        flags = split_data.map(lambda x: x[3]).distinct().collect()
        flagStr='","'.join(map(str,flags))

        print("data len "+str(len(protocols))+" "+str(len(services))+" "+str(len(flags)))

        print("protocol "+protstr)
        print("service "+servicesStr)
        print("flag "+flagStr)
        


    print "DONE!"













