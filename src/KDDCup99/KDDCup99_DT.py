#!/usr/bin/env python

import KDDCup99_common_ml as kup
from KDDCup99_common_ml import protocols, services, flags
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
 
    from pyspark.ml.feature import StandardScaler
    from pyspark.ml.linalg import Vectors
    
    from pyspark.ml.classification import OneVsRest
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator

    from pyspark.mllib.regression import LabeledPoint
    from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
    from pyspark.mllib.evaluation import BinaryClassificationMetrics
    from pyspark.mllib.evaluation import MulticlassMetrics

except ImportError as e:
    print ("Can not import Spark Modules", e)
    sys.exit(1)

def parse_as_fullTuple(orgFields,protocols,services,flags,trainType):
    #remove label 
    fields = orgFields[0:41]
    
    # convert protocol to numeric categorical variable
    curProtocol=fields[1]
    fields[1]=len(protocols)
    if curProtocol in protocols:
        fields[1] = protocols.index(curProtocol)
        
    # convert service to numeric categorical variable
    curService=fields[2]
    fields[2]=len(services)
    if curService in services:    
        fields[2] = services.index(curService)

    
    # convert flag to numeric categorical variable
    curflag=fields[3]
    fields[3]=len(flags)
    if curflag in flags:    
        fields[3] = flags.index(curflag)
  
    #multi class as default
    label=orgFields[41].replace(".","")
    label_num=len(kup.attack_names)
    if label in kup.attack_names:
        label_num=float(kup.attack_names.index(label)) 

    if trainType == "binary":
        # convert label to binary label
        label_num = 1.0
        if 'normal' in label:
            label_num = 0.0
        
    return LabeledPoint(label_num, array([float(x) for x in fields]))

def printUsageExit(status):
    print "Usage: spark-submit --driver-memory 4g KDDCup99_DT.py [option:train/test/all] trainingType(binary/multiClass) [treeDepth] [trainData] [testData]"
    sys.exit(status)

if __name__ == "__main__":
    if (len(sys.argv) < 6):
        printUsageExit(1)

    # set up environment
    opt=sys.argv[1]
    if opt == "help" :
        printUsageExit(0)
    trainType=sys.argv[2]
    treeDepth=int(sys.argv[3])
    data_file = sys.argv[4]    
    golden_file=sys.argv[5]

    if trainType != "binary" and trainType != "multiClass":
        print("trainType "+trainType)
        printUsageExit(1)

    spark = SparkSession.builder \
        .appName("KDDCup99_DT") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()

    sc = spark.sparkContext     
    log4jLogger = spark._jvm.org.apache.log4j 
    log4jLogger.LogManager.getLogger("org").setLevel(log4jLogger.Level.WARN)
    log4jLogger.LogManager.getLogger("akka").setLevel(log4jLogger.Level.OFF)

    dt_path=kup.saved_model_path+"/dtModel"
    if trainType == "multiClass":
        dt_path=kup.saved_model_path+"/dtMultiClassModel"
    
  

    if opt=="train" or opt =="all":

        # load raw data
        print "Loading RAW data..."
        raw_data = sc.textFile(data_file)
        split_data=raw_data.map(lambda x: x.split(","))
   
    
      
        print "Parsing dataset..."
        parsed_labelpoint = split_data.map(lambda x:parse_as_fullTuple(x,protocols,services,flags,trainType))
        
        # train a decision tree and save it as a file
        start=time()
        tree_model = DecisionTree.trainClassifier(parsed_labelpoint, numClasses=kup.num_attack_types, 
                                          categoricalFeaturesInfo={1: len(protocols), 2: len(services), 3: len(flags)},
                                          impurity='gini', maxDepth=treeDepth, maxBins=100,minInstancesPerNode=1)
        shutil.rmtree(dt_path, ignore_errors=True)
        tree_model.save(sc,dt_path)        
        trainTime=time()-start
        print("Train time: {} ".format(round(trainTime,3)))

    if opt=="test"  or opt=="all":

        #load test data
        test_split=sc.textFile(golden_file).map(lambda x:x.split(","))
        parsed_test=test_split.map(lambda x:parse_as_fullTuple(x,protocols,services,flags,trainType))

        # Evaluate model on test instances and compute test error
        model=DecisionTreeModel.load(sc,dt_path)
        start=time()
        predictions = model.predict(parsed_test.map(lambda x: x.features))
        predictions.count()                
        testTime=time()-start
        print("Test time: {} ".format(round(testTime,3)))

        # compute the f1 score on test data
        labelsAndPredictions = parsed_test.map(lambda lp: lp.label).zip(predictions)
        kup.computeF1ScoreForBinaryClassifier(labelsAndPredictions)
        print('Learned classification tree model:')
        print(model.toDebugString())

        






    print "DONE!"













