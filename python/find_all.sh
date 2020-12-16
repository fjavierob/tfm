#!/bin/bash

SAVE_BASE='./resultados1718/'
DATASET_DIR='../caracteristicas/'
PROGRESS_FILE='./progreso_find_all'

CONJUNTOS="1 2 3 4 5 6"
M=1
P1=3
P2=7

# Random Forest
MAX_DEPTH=('0' '30' '13' '27' '30' '20' '20')

export SAVE_BASE

if [ ! -d "$SAVE_BASE" ]; then
  mkdir -p $SAVE_BASE
fi

for C in $CONJUNTOS; do

    DATASET="dataset${C}_m${M}_${P1}_${P2}"

    echo "                     DATASET $DATASET"
    echo "------------------------------------------------------------------------------"

    DATASET_FILE="${DATASET_DIR}${DATASET}.csv"
    export DATASET_FILE

    # echo "$DATASET  ANN   $(date)" >> $PROGRESS_FILE
    # ./find_ann.sh

    # echo "$DATASET  KNN   $(date)" >> $PROGRESS_FILE
    # ./find_knn.sh

    # echo "$DATASET  SVM   $(date)" >> $PROGRESS_FILE
    # ./find_svm.sh

    echo "$DATASET  RF    $(date)" >> $PROGRESS_FILE
    export DEPTH_MAX=${MAX_DEPTH[$C]}
    ./find_rf.sh

done