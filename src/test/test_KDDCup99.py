import unittest


import sys
import os
from collections import OrderedDict
from numpy import array
from math import sqrt
spark_home = os.environ['SPARK_HOME']
sys.path.append(os.environ['kupDir'])
sys.path.insert(1, os.path.join(spark_home, 'python/lib/pyspark.zip'))
sys.path.append("/Users/minli/package/spark-2.1.1-bin-hadoop2.7/python")
sys.path.insert(1, os.path.join(spark_home, 'python/lib/py4j-0.10.4-src.zip'))
try:
    from pyspark import SparkContext, SparkConf
    from pyspark.sql import SparkSession
    from pyspark.sql.types import *
    
    from pyspark.mllib.clustering import KMeans
    from pyspark.ml.feature import StandardScaler
    from pyspark.mllib.evaluation import MulticlassMetrics
    from pyspark.ml.linalg import Vectors
    from pyspark.tests import ReusedPySparkTestCase
except ImportError as e:
    print ("Can not import Spark Modules", e)
    sys.exit(1)


import test_KDDCup99_common_ml
import test_KDDCup99_DT


print("test KDDCup99")
suite1 = test_KDDCup99_common_ml.TheTestSuite()
suite2 = test_KDDCup99_DT.TheTestSuite()
allsuites = unittest.TestSuite([suite1, suite2])
unittest.TextTestRunner(verbosity=2).run(allsuites)