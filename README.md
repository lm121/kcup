# Network Intrusion Detection
This project uses three algorithms including logistic regression, decision tree and kmeans to 
tackle the network intrusion competition in KDD Cup 1999 with python and Spark. 

### Documentation
```
docs
├── KDDCup99.pptx
├── experimentFigures.xlsx
```
KDDCup99.pptx provides a high level summary of the entire project. It discusses the statistics of the 
datasets, the algorithms used, and the parameter study and performance comparison between different
learning models.

experimentFigures.xlsx contains all the data and figures used in the KDDCup99.pptx document.

## Setup
Dependencies: Spark 2.1+, Python 2.7

Download the datasets from [here](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html).
Assume the dataset folder is stored in a folder named "datasets" at the same directory level of the kcup repository. 
If that is not the case, please change the "dataDir" variable in ./script/runKCup.sh


### Execution and unit test
```
scripts/
├── runAll.sh
└── runKCup.sh
```
This folder includes all the scripts needed to train and test the three different learning models, namely logistic regression, decision tree and kmeans.

To run logistic regression:
```
cd $kcup_HOME; ./scripts/runKcup.sh logRegBinary train/test/all [regularization parameter] 
```
To run decision tree:
```
cd $kcup_HOME; ./scripts/runKcup.sh DT train/test/all [binary/multiClass] [treeDepth] 
```
To run kmeans:
```
cd $kcup_HOME; ./scripts/runKcup.sh kmeans train/test/all [minNumClusters] [maxNumClusters]
```

The project also include several unit tests. To run unit tests:
```
cd $kcup_HOME; ./scripts/runKcup.sh test
```


### High level description of Source code
```
src
├── KDDCup99
│   ├── KDDCup99_common_ml.py
│   ├── KDDCup99_data_statistics.py
│   ├── KDDCup99_DT.py
│   ├── KDDCup99_kmeans.py
│   ├── KDDCup99_logreg_binary.py
│   ├── KDDCup99_logreg_multiclass.py
└── test
    ├── test_KDDCup99.py
    ├── test_KDDCup99_DT.py
    └── test_KDDCup99_common_ml.py
```

This folder contains the python source code implementing the training and testing of different models.
The file names are self explanatory except KDDCup99_common_ml.py which implements the common data preprocessing functions
used by most of other python modules.



