#!/bin/bash


curdir=$(dirname $0)
echo $curdir
exit 0

sh $curdir/runKCup.sh logRegBinary train
sh $curdir/runKCup.sh logRegBinary "test"
sh $curdir/runKCup.sh logRegBinary all

#DT
rm -f /tmp/dt.log;for ll in 6 8 10 12; do echo "tree depth $ll";./scripts/runKCup.sh DT all binary $ll;  done |tee /tmp/dt.log


#kmeans
rm -f /tmp/kmeans.log;for ll in 20 40 60 80 100 120; do echo "cluster size $ll";./scripts/runKCup.sh kmeans all $ll $ll;  done |tee /tmp/kmeans.log

rm -f /tmp/kmeans.log;for ll in 5 10; do echo "cluster size $ll";./scripts/runKCup.sh kmeans all $ll $ll;  done |tee /tmp/kmeans.log


#test logreg runtime
for((i=0;i<3;i++)); do ./scripts/runKCup.sh logRegBinary all 0.01; done  |tee /tmp/logregTime.log

#DT

for((i=0;i<4;i++)); do ./scripts/runKCup.sh DT all ; done  |tee /tmp/DTTime.log

#kmeans
for((i=0;i<4;i++)); do ./scripts/runKCup.sh kmeans test ; done  |tee /tmp/kmeansTime.log