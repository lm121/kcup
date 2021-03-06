#!/bin/bash

printUsageExit(){
	echo "$0: [algorithm:logRegBinary/logRegMulticlass/DT/kmeans/dataStat] [option:help/train/test/all]"
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
export kupDir=${curdir}/../src/KDDCup99
dataDir=${curdir}/../../datasets
testData=${dataDir}/corrected
trainData=${dataDir}/kddcup.data_10_percent.txt
#trainData=${dataDir}/kddcup.data.txt
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
	echo "execute DT; phase $phase; train type $trainType;treeDepth $treeDepth "
	time spark-submit --master ${sparkMaster} --total-executor-cores 4 --executor-memory 4g ${kupDir}/KDDCup99_DT.py $phase $trainType ${treeDepth} $trainData $testData

elif [  "$opt" = "kmeans" ];then

	if [ $# -ge 2 ];then phase=$2; fi
	min=24
	max=24
	if [ $# -ge 4 ];then min=$3;max=$4; fi
	echo "execute kmeans;  phase $phase; min $min ; max $max "
	time spark-submit --master ${sparkMaster} --total-executor-cores 4 --executor-memory 4g ${kupDir}/KDDCup99_kmeans.py $phase $min $max $trainData $testData

elif [ $opt == "dataStat" ];then
	
	if [ $# -ge 2 ];then phase=$2; fi
	echo "compute data statistics; phase $phase "
	time spark-submit --master ${sparkMaster} --total-executor-cores 4 --executor-memory 4g ${kupDir}/KDDCup99_data_statitics.py $phase $trainData $testData

elif [[ $opt = "trySpark" ]];then
	echo "explore spark"
	time spark-submit --master ${sparkMaster} --total-executor-cores 4 --executor-memory 4g ${kupDir}/explore_spark.py  $trainData $testData

elif [ "$opt" = "test" ]; then
	testDir="${curdir}/../src/test"

	echo "kupdir $kupDir;testDir $testDir "
	
	cd $testDir
	time python -m unittest test_KDDCup99

else
	printUsageExit
fi

exit 0
