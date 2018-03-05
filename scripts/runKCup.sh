#!/bin/bash

printUsageExit(){
	echo "$0: [algorithm:logRegBinary/logRegMulticlass/DT/kmeans/dataStat] [option:train/test/all]"
	exit 0
}
if [ $# -lt 1 ]; then
	printUsageExit
fi

curdir=$(dirname $0)
cd $curdir;
curdir=$(pwd)


#define variables needed in this script
sparkMaster=local[*]
export saved_model_path="$curdir/../trained_model"
kupDir=${curdir}/../src/KDDCup99
dataDir=${curdir}/../../datasets
testData=${dataDir}/corrected
trainData=${dataDir}/kddcup.data_10_percent.txt
opt=$1
phase="help"

# note the bash script should go to the directory storing the python scripts
# the python code import some files within the same directory
cd $kupDir

if [ $opt == "logRegBinary" ];then	
	if [ $# -ge 2 ];then phase=$2 ; fi
	echo "execute logistic regression binary class;  phase $phase; "
	regParam=0.01
	if [ $# -ge 3 ];then regParam=$3; fi
	time spark-submit --master ${sparkMaster} --total-executor-cores 4 --executor-memory 4g ${kupDir}/KDDCup99_logreg_binary.py  $phase $regParam $trainData $testData

elif  [ $opt == "logRegMulticlass" ];then

	if [ $# -ge 2 ];then phase=$2; fi
	echo "execute logistic regression multiclassifier; phase $phase; "
	time spark-submit --master ${sparkMaster} --total-executor-cores 4 --executor-memory 4g ${kupDir}/KDDCup99_logreg_multiclass.py $phase $trainData $testData


elif [ $opt == "DT" ];then
	
	if [ $# -ge 2 ];then phase=$2; fi
	trainType="binary"
	if [ $# -ge 3 ];then trainType=$3; fi
	treeDepth=4
	if [ $# -ge 4 ];then treeDepth=$4; fi
	echo "execute DT phase; $phase $trainType;treeDepth $treeDepth "
	time spark-submit --master ${sparkMaster} --total-executor-cores 4 --executor-memory 4g ${kupDir}/KDDCup99_DT.py $phase $trainType ${treeDepth} $trainData $testData

elif [  "$opt" = "kmeans" ];then

	if [ $# -ge 2 ];then phase=$2; fi
	min=40
	max=40
	if [ $# -ge 4 ];then min=$3;max=$4; fi
	echo "execute kmeans;  phase $phase; min $min ; max $max "
	time spark-submit --master ${sparkMaster} --total-executor-cores 4 --executor-memory 4g ${kupDir}/KDDCup99_kmeans.py $phase $min $max $trainData $testData

elif [ $opt == "dataStat" ];then
	
	if [ $# -ge 2 ];then phase=$2; fi
	trainType="binary"
	if [ $# -ge 3 ];then trainType=$3; fi
	echo "execute DT; phase $phase $trainType"
	time spark-submit --master ${sparkMaster} --total-executor-cores 4 --executor-memory 4g ${kupDir}/KDDCup99_DT.py $phase $trainType $trainData $testData

elif [[ $opt = "trySpark" ]];then
	echo "explore spark"
	time spark-submit --master ${sparkMaster} --total-executor-cores 4 --executor-memory 4g ${kupDir}/explore_spark.py  $trainData $testData

elif [ "$opt" = "test1" ]; then
	echo "test1 kupdir $kupDir "
	echo "model path: $saved_model_path"


else
	printUsageExit
fi

exit 0
