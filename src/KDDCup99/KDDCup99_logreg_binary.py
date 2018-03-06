#!/usr/bin/env python


import KDDCup99_common_ml as kup
import sys
import os
import shutil
from time import time
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
          "KDDCup99_logreg_binary.py [option:train/test/all] [regularizationParameter] [trainData] [testData]"
    sys.exit(status)

if __name__ == "__main__":
    if (len(sys.argv) < 5):
        printUsageExit(1)

    # set up environment
    opt=sys.argv[1]
    if opt == "help" :
        printUsageExit(0)
    regularizationParameter=float(sys.argv[2])
    data_file = sys.argv[3]
    golden_file=sys.argv[4]
    model_path=kup.saved_model_path+"/logregModel"
    scaler_model_path=kup.saved_model_path+"/scalerModel"

    spark = SparkSession \
        .builder \
        .appName("KDDCup99_logreg") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()

    sc = spark.sparkContext       
    log4jLogger = spark._jvm.org.apache.log4j 
    log4jLogger.LogManager.getLogger("org").setLevel(log4jLogger.Level.WARN)
    log4jLogger.LogManager.getLogger("akka").setLevel(log4jLogger.Level.OFF)

    
    if opt == "train" or opt== "all":
        # load raw data
        print "Loading RAW data..."
        raw_data = sc.textFile(data_file)


        
        print "Parsing dataset..."
        parsed_labelpoint = raw_data.map(kup.parse_as_binaryTuple).filter(lambda x : x[0]!=-1.0)
        parsed_labelpoint_df=spark.createDataFrame(parsed_labelpoint,["label","features"])

        print "Standardizing data..."
        scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                            withStd=True, withMean=True)

        # Feature scaling with  StandardScaler
        scalerModel = scaler.fit(parsed_labelpoint_df)
        shutil.rmtree(scaler_model_path, ignore_errors=True)
        scalerModel.save(scaler_model_path)

        # Normalize each feature to have unit standard deviation.
        train_df_tmp= scalerModel.transform(parsed_labelpoint_df)
        train_df=train_df_tmp.drop("features").withColumnRenamed("scaledFeatures","features")
        # count labels
        tmp_df=train_df.groupBy("label").count()
        tmp_df.show(10)

        start=time()       
        lr = LogisticRegression(maxIter=20, regParam=regularizationParameter, elasticNetParam=0.2,family="binomial")

        # Fit and save the model
        lrModel = lr.fit(train_df)
        shutil.rmtree(model_path)
        lrModel.save(model_path)
        trainTime=time()-start
        print("Train time: {} ".format(round(trainTime,3)))
        # print model summary
        trainingSummary = lrModel.summary            
        print("areaUnderROC: " + str(trainingSummary.areaUnderROC))
        fMeasure = trainingSummary.fMeasureByThreshold
        maxFMeasure = fMeasure.groupBy().max('F-Measure').select('max(F-Measure)').head()
        print("Fmeasure "+str(maxFMeasure))

        
    if opt=="test" or opt=="all":
        
        lrModel=LogisticRegressionModel.load(model_path)

        # load test data and perform feature scaling on test data
        test_data=sc.textFile(golden_file)  
        parsed_test_data=test_data.map(kup.parse_as_binaryTuple).filter(lambda x : x[0]!=-1.0)   
        parsed_test_data_df=spark.createDataFrame(parsed_test_data,["label","features"])

        scalerModel=StandardScalerModel.load(scaler_model_path)
        test_df_tmp= scalerModel.transform(parsed_test_data_df)
        test_df=test_df_tmp.drop("features").withColumnRenamed("scaledFeatures","features")
        start=time()
        testPrediction= lrModel.transform(test_df)            
        testPrediction.show(3)
        testTime=time()-start
        print("Test time: {} ".format(round(testTime,3)))
        testPrediction.printSchema()
        predictionAndLabelsRdd=testPrediction.select("label","prediction").rdd            
        kup.computeF1ScoreForBinaryClassifier(predictionAndLabelsRdd)


        

    print "DONE!"













