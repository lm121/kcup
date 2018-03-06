#!/usr/bin/env python



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
    from pyspark.ml.feature import StandardScaler
    from pyspark.mllib.evaluation import MulticlassMetrics
    from pyspark.ml.linalg import Vectors
    
except ImportError as e:
    print ("Can not import Spark Modules", e)
    sys.exit(1)



saved_model_path=os.environ["saved_model_path"]

attack_types={'normal':0,'back':1,'buffer_overflow':2,'ftp_write':3,'guess_passwd':4,'imap':5,'ipsweep':6,'land':7,'loadmodule':8,'multihop':9,'neptune':10,'nmap':11,'perl':12,'phf':13,'pod':14,'portsweep':15,'rootkit':16,'satan':17,'smurf':18,'spy':19,'teardrop':20,'warezclient':21,'warezmaster':22}
num_attack_types=len(attack_types)+1 

#all the names of attacks within the datasets
attack_names=['normal','back','buffer_overflow','ftp_write','guess_passwd','imap','ipsweep','land','loadmodule','multihop','neptune','nmap','perl','phf','pod','portsweep','rootkit','satan','smurf','spy','teardrop','warezclient','warezmaster']

# the column name of each field, label is the last column
col_names = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]

# this arrays stores the columns with numerical values, 4 fields are removed, namely: protocol_type, service,flag and label
num_features = [
    "duration","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate"
]


# the unique values of categorical fields in the dataset with data length 3, 66,11.
protocols=["icmp","tcp","udp"]
services=["supdup","domain","http","whois","netstat","netbios_dgm","time","smtp","gopher","private","name","ctf","ntp_u","red_i","vmnet","tim_i","klogin","nnsp","mtp","hostnames","pop_2","urp_i","Z39_50","eco_i","ftp_data","pm_dump","systat","ftp","sql_net","efs","remote_job","finger","ldap","kshell","iso_tsap","ecr_i","nntp","printer","telnet","uucp","auth","http_443","tftp_u","login","echo","sunrpc","urh_i","uucp_path","daytime","other","pop_3","netbios_ns","shell","domain_u","courier","exec","rje","link","ssh","netbios_ssn","csnet_ns","X11","IRC","bgp","discard","imap4"]
flags=["S2","RSTR","OTH","RSTO","S1","SH","RSTOS0","S3","REJ","S0","SF"]


def attacktypeToNumerical(att):
    """
    convert a string of label to either 0.0 or 1.0.
    a value of 0 represents normal, a value of 1 represents abnormal
    """
    if  "normal" in att:
        return 0.0
    return 1.0

def parse_interaction(line):
    """
    Parses a network data instance.
    return a tuple (label, feature array)
    """
    line_split = line.split(",")
    clean_line_split = [line_split[0]]+line_split[4:-1]
    return (line_split[-1], array([float(x) for x in clean_line_split]))

def parse_multiClass(line):
    """
    convert the label to numerical value, based on the index of the label with 
    the array attack_names
    """
    line_split = line.split(",")
    clean_line_split = [line_split[0]]+line_split[4:-1]
    label=line_split[-1].replace(".","")
    label_num=len(attack_names)
    if label in attack_names:
        label_num=float(attack_names.index(label))    
    return (label_num, Vectors.dense(array([float(x) for x in clean_line_split])))

def parse_as_binaryTuple(line):
    """
    parse one line as a tuple, with label as 0.0(normal) or 1.0(abnormal)
    """
    line_split = line.split(",")
    clean_line_split = [line_split[0]]+line_split[4:-1]
    label=line_split[-1].replace(".","")
    label_num=attacktypeToNumerical(label)
    try:
        labelfeature=(label_num, Vectors.dense(array([float(x) for x in clean_line_split])))
    except:
        labelfeature=(-1.0,Vectors.dense(array(range(0,len(clean_line_split)))))
    return labelfeature



def countingLabel(raw_data):
    """
    given the raw data from the original training data,
    extract out the labels, count the frequency of each label
    and print out the label and its frequency
    """
    print ("Counting all different labels")
    labels = raw_data.map(lambda line: line.strip().split(",")[-1])
    label_counts = labels.countByValue()
    sorted_labels = OrderedDict(sorted(label_counts.items(), key=lambda t: t[1], reverse=True))
    for label, count in sorted_labels.items():
        print( str(label)+" "+ str(count))

def computeF1ScoreForBinaryClassifier(predictLabelRdd):
    """
    compute the F1 Score given a rdd with each line representing a tuple of (label,predict)
    it also computes the precision, recall, accuracy and test error.
    it is used to compare with the result of metric modules provided by Spark.
    """
    tp=predictLabelRdd.filter(lambda (v, p): v == 1.0 and p==1.0 ).count() 
    tn=predictLabelRdd.filter(lambda (v, p): v == 0.0 and p==0.0 ).count() 
    fp=predictLabelRdd.filter(lambda (v, p): v == 0.0 and p==1.0 ).count() 
    fn=predictLabelRdd.filter(lambda (v, p): v == 1.0 and p==0.0 ).count() 
    prec=float(tp)/(tp+fp)
    recall=float(tp)/(tp+fn)
    f1=(2*prec*recall)/(prec+recall)
    testErr = predictLabelRdd.filter(lambda (v, p): v != p).count() / float(predictLabelRdd.count())

    print("confusion matrix: "+str(tp)+" "+str(tn)+" "+str(fp)+" "+str(fn))
    print("Precision Recall f1: "+str(prec)+" "+str(recall)+" "+str(f1))
    print('Test Error = ' + str(testErr) + ' Accuracy = '+str(1-testErr))
    return(prec,recall,f1,testErr)

def computeF1ScoreForMultiClassifier(predictLabelRdd):
    """
    compute the F1 Score given a rdd with each line representing a tuple of (label,predict)
    the labels are multi-class labels with number of unique values greater than 2.
    it also computes the precision, recall, accuracy and test error.
    it is used to compare with the result of metric modules provided by Spark.
    This is used for kmeans to check whether the kmeans model successfully differentiate normal connections 
    from abnormal ones.
    """
    tp=predictLabelRdd.filter(lambda (v, p): v >= 1.0 and p>=1.0 ).count() 
    tn=predictLabelRdd.filter(lambda (v, p): v == 0.0 and p==0.0 ).count() 
    fp=predictLabelRdd.filter(lambda (v, p): v == 0.0 and p>=1.0 ).count() 
    fn=predictLabelRdd.filter(lambda (v, p): v >= 1.0 and p==0.0 ).count() 
    prec=float(tp)/(tp+fp)
    recall=float(tp)/(tp+fn)
    f1=(2*prec*recall)/(prec+recall)
    testErr = predictLabelRdd.filter(lambda (v, p): v != p).count() / float(predictLabelRdd.count())

    print("confusion matrix: "+str(tp)+" "+str(tn)+" "+str(fp)+" "+str(fn))
    print("prec Recall f1: "+str(prec)+" "+str(recall)+" "+str(f1))
    print("Test Error = " + str(testErr) + ' Accuracy = '+str(1-testErr))
    return(prec,recall,f1,testErr)

