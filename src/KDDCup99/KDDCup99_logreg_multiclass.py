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
    from pyspark.ml.classification import LogisticRegression

    from pyspark.ml.classification import OneVsRest,OneVsRestModel
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator

    
    from pyspark.mllib.evaluation import RegressionMetrics
    from pyspark.mllib.evaluation import MulticlassMetrics
except ImportError as e:
    print ("Can not import Spark Modules", e)
    sys.exit(1)

def printUsageExit(status):
    print "Usage: spark-submit --driver-memory 4g " + \
          "KDDCup99_logreg_multiclass.py [option:train/test/all] [trainData] [testData]"
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

    scalerPath=kup.saved_model_path+"/scalerMultiModel"
    logreg_path=kup.saved_model_path+"/logregMultiModel"

    spark = SparkSession \
        .builder \
        .appName("KDDCup99_logreg") \
        .config("spark.executor.memory", "2g") \
        .getOrCreate()

    sc = spark.sparkContext
    log4jLogger = spark._jvm.org.apache.log4j 
    log4jLogger.LogManager.getLogger("org").setLevel(log4jLogger.Level.WARN)
    log4jLogger.LogManager.getLogger("akka").setLevel(log4jLogger.Level.OFF)



    if opt=="train" or opt =="all":

        
        print "Loading RAW data..."
        raw_data = sc.textFile(data_file)
      
        print "Parsing dataset..."
        parsed_labelpoint = raw_data.map(kup.parse_multiClass)
        parsed_labelpoint_df=spark.createDataFrame(parsed_labelpoint,["label","features"])
    
       
        print "Standardizing data..."
        scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                            withStd=True, withMean=True)

        # train a scaler to perform feature scaling
        scalerModel = scaler.fit(parsed_labelpoint_df)
        shutil.rmtree(scalerPath,ignore_errors=True)
        scalerModel.save(scalerPath)

        # Normalize each feature to have unit standard deviation.
        train_df_tmp= scalerModel.transform(parsed_labelpoint_df)
        train_df=train_df_tmp.drop("features").withColumnRenamed("scaledFeatures","features")

        # show the frequency of each label
        tmp_df=train_df.groupBy("label").count()
        tmp_df.show(10)
        
        

        lr = LogisticRegression(maxIter=10, regParam=0.01, elasticNetParam=0.2)

        # instantiate the One Vs Rest Classifier.
        ovr = OneVsRest(classifier=lr)
        # train the multiclass model.
        lrModel = ovr.fit(train_df)
        shutil.rmtree(logreg_path, ignore_errors=True)
        lrModel.save(logreg_path)

    if opt=="test" or opt =="all":

        print "Loading test data..."
        test_data=sc.textFile(golden_file)  
        parsed_test_data=test_data.map(kup.parse_multiClass)   
        parsed_test_data_df=spark.createDataFrame(parsed_test_data,["label","features"])

        # load the scaler and perform feature scaling on test data
        scalerModel=StandardScalerModel.load(scalerPath)
        test_df_tmp= scalerModel.transform(parsed_test_data_df)
        test_df=test_df_tmp.drop("features").withColumnRenamed("scaledFeatures","features")

        print "Predicting test data..."
        lrModel=OneVsRestModel.load(logreg_path)
        testPrediction= lrModel.transform(test_df) 

        testPrediction.printSchema()
        testPrediction.show(5)

        print "eval..."

        evaluator = MulticlassClassificationEvaluator(metricName="accuracy")

        # compute the classification error on test data.
        accuracy = evaluator.evaluate(testPrediction)
        print("Test Error = %g" % (1.0 - accuracy))
        
        predictionAndLabels=testPrediction.select("label","prediction").rdd
        
        #compute a simplified f1 score on prediction compared with labels of test data
        kup.computeF1ScoreForMultiClassifier(predictionAndLabels) 

        # compute precision, recall and f1 using multiclassMetrics from Spark
        metrics = MulticlassMetrics(predictionAndLabels)        
        precision = metrics.precision()
        recall = metrics.recall()
        f1Score = metrics.fMeasure()
        print("Summary Stats")
        print("Precision = %s" % precision)
        print("Recall = %s" % recall)
        print("F1 Score = %s" % f1Score)


    print "DONE!"













