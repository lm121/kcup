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
    from pyspark.mllib.regression import LabeledPoint
    from pyspark.mllib.clustering import KMeans
    from pyspark.ml.feature import StandardScaler
    from pyspark.mllib.evaluation import MulticlassMetrics
    from pyspark.ml.linalg import Vectors
    from pyspark.tests import ReusedPySparkTestCase
except ImportError as e:
    print ("Can not import Spark Modules", e)
    sys.exit(1)


from KDDCup99_DT import *
from KDDCup99_common_ml import *

def TheTestSuite():
    return unittest.TestLoader().loadTestsFromTestCase(TestDT)

class TestDT(ReusedPySparkTestCase):
	


    def test_parse_as_fullTuple(self):
        line="0,tcp,http,SF,212,786,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,8,8,0.00,0.00,0.00,0.00,1.00,0.00,0.00,8,99,1.00,0.00,0.12,0.05,0.00,0.00,0.00,0.00,normal.".split(",")
        expected=LabeledPoint(0.0,array([0.0,1,2,10,212.0,786.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,8.0,8.0,0.00,0.00,0.00,0.00,1.00,0.00,0.00,8.0,99.0,1.00,0.00,0.12,0.05,0.00,0.00,0.00,0.00]))
        parsed_line=parse_as_fullTuple(line,protocols,services,flags,"binary")

        self.assertTrue(expected.label==parsed_line.label)
        self.assertTrue(expected.features==parsed_line.features)


        line="0,udp,private,SF,105,146,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,0.00,0.00,0.00,0.00,1.00,0.00,0.00,255,254,1.00,0.01,0.00,0.00,0.00,0.00,0.00,0.00,neptune.".split(",")
        expected=LabeledPoint(1.0,array([0.0,2,9,10,105.0,146.0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,0.00,0.00,0.00,0.00,1.00,0.00,0.00,255,254,1.00,0.01,0.00,0.00,0.00,0.00,0.00,0.00]))

        parsed_line=parse_as_fullTuple(line,protocols,services,flags,"binary")
        self.assertTrue(expected.label==parsed_line.label)
        self.assertTrue(expected.features==parsed_line.features)
      

if __name__ == '__main__':
    unittest.main()