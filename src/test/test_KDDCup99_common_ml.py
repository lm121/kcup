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


from KDDCup99_common_ml import *

def TheTestSuite():
    return unittest.TestLoader().loadTestsFromTestCase(TestCommonML)

class TestCommonML(ReusedPySparkTestCase):

    def test_parse_interaction(self):
        line="0,tcp,http,SF,212,786,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,8,8,0.00,0.00,0.00,0.00,1.00,0.00,0.00,8,99,1.00,0.00,0.12,0.05,0.00,0.00,0.00,0.00,normal."
        expected=("normal.",array([0.0,212.0,786.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,8.0,8.0,0.00,0.00,0.00,0.00,1.00,0.00,0.00,8.0,99.0,1.00,0.00,0.12,0.05,0.00,0.00,0.00,0.00]))
        parsed_line=parse_interaction(line)

        self.assertTrue(expected[0]==parsed_line[0])
        self.assertTrue((expected[1]==parsed_line[1]).all())

    def test_parse_multiClass(self):
        line="0,tcp,http,SF,212,786,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,8,8,0.00,0.00,0.00,0.00,1,0.00,0.00,8,99,1,0.00,0.12,0.05,0.00,0.00,0.00,0.00,normal."
        expected=(0.0,Vectors.dense(array([0.0,212.0,786.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,8.0,8.0,0.00,0.00,0.00,0.00,1.00,0.00,0.00,8.0,99.0,1.00,0.00,0.12,0.05,0.00,0.00,0.00,0.00])))
        parsed_line=parse_multiClass(line)
        self.assertTrue(expected[0]==parsed_line[0])
        self.assertTrue((expected[1]==parsed_line[1]))

        line="0,udp,private,SF,105,146,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,0.00,0.00,0.00,0.00,1.00,0.00,0.00,255,254,1.00,0.01,0.00,0.00,0.00,0.00,0.00,0.00,neptune."
        expected=(10.0,Vectors.dense(array([0.0,105.0,146.0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,0.00,0.00,0.00,0.00,1.00,0.00,0.00,255,254,1.00,0.01,0.00,0.00,0.00,0.00,0.00,0.00])))

        parsed_line=parse_multiClass(line)
        self.assertTrue(expected[0]==parsed_line[0])
        self.assertTrue((expected[1]==parsed_line[1]))



    def test_parse_as_binaryTuple(self):
        line="0,tcp,http,SF,212,786,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,8,8,0.00,0.00,0.00,0.00,1.00,0.00,0.00,8,99,1.00,0.00,0.12,0.05,0.00,0.00,0.00,0.00,normal."
        expected=(0.0,Vectors.dense(array([0.0,212.0,786.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,8.0,8.0,0.00,0.00,0.00,0.00,1.00,0.00,0.00,8.0,99.0,1.00,0.00,0.12,0.05,0.00,0.00,0.00,0.00])))
        parsed_line=parse_as_binaryTuple(line)

        self.assertTrue(expected[0]==parsed_line[0])
        self.assertTrue((expected[1]==parsed_line[1]))

        line="0,udp,private,SF,105,146,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,0.00,0.00,0.00,0.00,1.00,0.00,0.00,255,254,1.00,0.01,0.00,0.00,0.00,0.00,0.00,0.00,neptune."
        expected=(1.0,Vectors.dense(array([0.0,105.0,146.0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,0.00,0.00,0.00,0.00,1.00,0.00,0.00,255,254,1.00,0.01,0.00,0.00,0.00,0.00,0.00,0.00])))

        parsed_line=parse_as_binaryTuple(line)
        self.assertTrue(expected[0]==parsed_line[0])
        self.assertTrue((expected[1]==parsed_line[1]))



    def test_computeF1ScoreForBinaryClassifier(self):
    	prediction1 = self.sc.parallelize([(1, 0),(0,0),(1,0),(1,1)])
    	#TODO:Add test
    	self.assertTrue(True)

    def test_computeF1ScoreForMultiClassifier(self):
		prediction1 = self.sc.parallelize([(1, 0),(0,0),(1,0),(1,1)])
		#TODO: add test
		self.assertTrue(True)



if __name__ == '__main__':
    unittest.main()


