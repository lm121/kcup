#!/usr/bin/env python

# dataset: http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html

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
    from pyspark.ml.clustering import KMeans,KMeansModel
    from pyspark.ml.feature import StandardScaler,StandardScalerModel
    from pyspark.sql import SparkSession
    from pyspark.sql.types import *
    import pyspark.sql.functions as F
    from pyspark.sql.functions import col,struct
    from pyspark.mllib.evaluation import MulticlassMetrics

    
    print ("Successfully imported Spark Modules")
except ImportError as e:
    print ("Can not import Spark Modules", e)
    sys.exit(1)





def distance(a, b):
    """
    Calculates the euclidean distance between two numeric RDDs
    """
    return sqrt(
        a.zip(b)
        .map(lambda x: (x[0]-x[1]))
        .map(lambda x: x*x)
        .reduce(lambda a,b: a+b)
        )

def dist_to_centroid(datum, clusters):
    """
    Determines the distance of a point to its cluster centroid
    """
    cluster = clusters.predict(datum)
    centroid = clusters.centers[cluster]
    return sqrt(sum([x**2 for x in (centroid - datum)]))


def clustering_score(data, k):
    clusters = KMeans.train(data, k, maxIterations=10, runs=5, initializationMode="random")
    result = (k, clusters, data.map(lambda datum: dist_to_centroid(datum, clusters)).mean())
    print "Clustering score for k=%(k)d is %(score)f" % {"k": k, "score": result[2]}
    return result

def clustering_score_vec(data,ink):    
    kmeans = KMeans(k=ink,seed=1,maxIter=20)
    print("training kmeansML model")
    model= kmeans.fit(data)       
    cost = model.computeCost(data)    
    result =(ink, model, cost)
    print "Clustering score for k=%(k)d is %(score)f" % {"k": ink, "score": result[2]}
    return result

def printUsageExit(status):
    print "Usage: spark-submit --driver-memory 4g " + \
          "KDDCup99_kmenas.py [option:train/test/all] [trainData] [testData]"
    sys.exit(status)

if __name__ == "__main__":
    if (len(sys.argv) < 5):
        print "Usage: spark-submit --driver-memory 2g KDDCup99.py min_k max_k [trainData] [testData]"
        sys.exit(1)

    # set up environment
    opt=sys.argv[1]
    min_k= int(sys.argv[2])
    max_k = int(sys.argv[3])
    data_file = sys.argv[4]
    golden_file=sys.argv[5]

    model_path=kup.saved_model_path+"/kmeansMLModel"
    scaler_model_path=kup.saved_model_path+"/scalerKmeansMLModel"
    cluster_label_path=kup.saved_model_path+"/kmeansML_cluster_label.csv"
    spark = SparkSession \
        .builder \
        .appName("KDDCup99_kmeans") \
        .config("spark.executor.memory", "2g") \
        .getOrCreate()

    sc = spark.sparkContext


    log4jLogger = spark._jvm.org.apache.log4j 
    log4jLogger.LogManager.getLogger("org").setLevel(log4jLogger.Level.WARN)
    log4jLogger.LogManager.getLogger("akka").setLevel(log4jLogger.Level.OFF)

    if opt=="train" or opt=="all":
                # load raw data
        print "Loading RAW data..."
        raw_data = sc.textFile(data_file)

        print "Parsing dataset..."
        parsed_labelpoint = raw_data.map(kup.parse_multiClass)
        parsed_labelpoint_df=spark.createDataFrame(parsed_labelpoint,["label","features"])
        parsed_labelpoint_df.cache()
        print "Standardizing data..."

        scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                            withStd=True, withMean=True)

        # Compute summary statistics by fitting the StandardScaler
        scalerModel = scaler.fit(parsed_labelpoint_df)

        shutil.rmtree(scaler_model_path, ignore_errors=True)
        scalerModel.save(scaler_model_path)
        # Normalize each feature to have unit standard deviation.
        train_df_tmp= scalerModel.transform(parsed_labelpoint_df)
        train_df=train_df_tmp.drop("features").withColumnRenamed("scaledFeatures","features")
        train_df.cache()
        tmp_df=train_df.groupBy("label").count()
        tmp_df.show(10)
        train_df.printSchema()
        train_df.show(3)

        start=time()
        # Evaluate values of k from min_k to max_k+1
        print "Calculating total in within cluster distance for different k values (%(min_k)d to %(max_k)d):" % {"min_k":min_k,"max_k": max_k}
        scores = map(lambda k: clustering_score_vec(train_df, k), range(min_k,max_k+1,10))

        # Obtain min score k
        min_k = min(scores, key=lambda x: x[2])[0]
        print "Best k value is %(best_k)d" % {"best_k": min_k}

        # Use the best model to assign a cluster to each datum
        # We use here standardized data - it is more appropriate for exploratory purposes
        print "Obtaining clustering result sample for k=%(min_k)d..." % {"min_k": min_k}
        best_model = min(scores, key=lambda x: x[2])[1]
        shutil.rmtree(model_path, ignore_errors=True)
        best_model.save(model_path)        
        trainTime=time()-start
        print("Train time: {} ".format(round(trainTime,3)))

        print("summary type" +str(type(best_model.summary.cluster)))
        best_model.summary.cluster.printSchema()

        # compute the corresponding attack type for each cluster
        predict_df=best_model.transform(train_df)
        cntdf=predict_df.groupBy(["prediction","label"]).count().sort(col("prediction"))
        cluster_label=cntdf.groupBy("prediction").agg(F.max(struct(col("count"), col("label"))).alias("max")).select("prediction","max.label")
        cluster_label.write.csv(cluster_label_path,header='true', mode='overwrite')
        tmpdf=cluster_label.groupBy("label").count()       
        print("covered number of attack types "+str(tmpdf.count()))
        tmpdf.show(50)

    if opt=="test" or opt =="all":
        # evaluation
        print "evaluation"

        
        cluster_label=spark.read.csv(cluster_label_path,header='true',inferSchema='true')
        cluster_label.printSchema()
        cluster_label.show(5)
        #sys.exit(0)
        test_data=sc.textFile(golden_file)  
        parsed_test_data=test_data.map(kup.parse_as_labelpoint)   
        parsed_test_data_df=spark.createDataFrame(parsed_test_data,["label","features"])

        scalerModel=StandardScalerModel.load(scaler_model_path)
        test_df_tmp= scalerModel.transform(parsed_test_data_df)
        test_df=test_df_tmp.drop("features").withColumnRenamed("scaledFeatures","features")

        best_model=KMeansModel.load(model_path)
        start=time()
        #predict_df=best_model.transform(test_df).select("label".alias("actualLabel"),"prediction".alias("predic"))
        predict_df=best_model.transform(test_df).select(col("label").alias("actualLabel"),"prediction")
        
        labelPredictedLabel=predict_df.join(cluster_label,cluster_label.prediction== predict_df.prediction).select(predict_df.actualLabel, cluster_label.label) 
        labelPredictedLabel.show(3)
        #labelPredictedLabel.printSchema()    
        testTime=time()-start
        print("Test time: {} ".format(round(testTime,3)))
        predictionAndLabels=labelPredictedLabel.rdd


        

        errCnt=predictionAndLabels.filter(lambda (v, p): v != p).count()
        trainErr =  errCnt/ float(parsed_test_data.count())
        print("Training Error = " + str(trainErr)+ " accuracy "+str(1-trainErr))

        kup.computeF1ScoreForMultiClassifier(predictionAndLabels)
        metrics = MulticlassMetrics(predictionAndLabels)
        # Overall statistics
        precision = metrics.precision()
        recall = metrics.recall()
        f1Score = metrics.fMeasure()
        print("Summary Stats")
        print("Precision = %s" % precision)
        print("Recall = %s" % recall)
        print("F1 Score = %s" % f1Score)


    if opt == "assignLabelToCluster":
        # analyze the class type; assign type to clusters
        raw_data = sc.textFile(data_file)
        parsed_labelpoint = raw_data.map(kup.parse_multiClass)
        parsed_labelpoint_df=spark.createDataFrame(parsed_labelpoint,["label","features"])
        scalerModel=StandardScalerModel.load(scaler_model_path)
        train_df_tmp= scalerModel.transform(parsed_labelpoint_df)
        train_df=train_df_tmp.drop("features").withColumnRenamed("scaledFeatures","features")

        best_model=KMeansModel.load(model_path)
        predict_df=best_model.transform(train_df)

        cntdf=predict_df.groupBy(["prediction","label"]).count().sort(col("prediction"))
        cluster_label=cntdf.groupBy("prediction").agg(F.max(struct(col("count"), col("label"))).alias("max")).select("prediction","max.label")

        cluster_label.write.csv(cluster_label_path,header='true', mode='overwrite')
        
        cluster_label.printSchema()
        cluster_label.show(50)
        
  

    if opt=="saveSampleAssignment":
        # Save assignment sample to file

        cluster_assignments_sample = standardized_data_values.map(lambda datum: str(best_model.predict(datum))+","+",".join(map(str,datum))).sample(False,0.05)

        print "Saving sample to file..."
        cluster_assignments_sample.saveAsTextFile("sample_standardized")

    print "DONE!"
