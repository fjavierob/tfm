#!/bin/bash

ENV_="tfm36"
PYTHON_FILE="clasificador_svm.py"
RESULTS_TMP="./svm_tmp"
HIGH_ACC=70

# DATASET_DIR='../caracteristicas/'
# DATASET_NAME='dataset2_m1_3_7.csv'
# DATASET_FILE="${DATASET_DIR}""${DATASET_NAME}"

DATASET_NAME=$(echo ${DATASET_FILE##*/} | cut -f1 -d.)

# SAVE_BASE="./resultados/"
SAVE_DIR="${SAVE_BASE}svm/${DATASET_NAME}"
RESULTS_FILE="${SAVE_DIR}/resultados.txt"

# Parametros SVM
KERNELS="linear poly rbf sigmoid"
C="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9"
SEED_MIN=0
SEED_MAX=15


SEED=$(eval echo "{$SEED_MIN..$SEED_MAX}")

if [[ "$CONDA_DEFAULT_ENV" != "$ENV_" ]]; then
    activate $ENV_
fi
if [ ! -d "$SAVE_DIR" ]; then
  mkdir -p $SAVE_DIR
fi

echo "" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE
echo "          ${DATASET_FILE##*/} $(date)" >> $RESULTS_FILE
echo "------------------------------------------------------------------------------------------------------" >> $RESULTS_FILE

for kernel in $KERNELS; do
    echo "kernel = $kernel"
    for c in $C; do
        echo -n "  C=${c}, Seed: "
        for seed in $SEED; do
            echo -n "$seed "
            # Ejecutar KNN
            python $PYTHON_FILE $DATASET_FILE $seed $c $kernel $RESULTS_TMP > /dev/null 2>&1
            # Resultado con datos escalados
            RESULTADO_VAL=$(cat $RESULTS_TMP | cut -f1 -d,)
            RESULTADO_ALL=$(cat $RESULTS_TMP | cut -f2 -d,) 
            RESULTADO_VAL_FLOOR="${RESULTADO_VAL:0:2}"
            RESULTADO_ALL_FLOOR="${RESULTADO_ALL:0:2}"
            if [[ RESULTADO_VAL_FLOOR -ge HIGH_ACC ]]; then
                if [[ RESULTADO_ALL_FLOOR -ge HIGH_ACC ]]; then
                    echo "[!!!]  ${RESULTADO_VAL} - ${RESULTADO_ALL} : C=${c}, kernel=${kernel}, seed=${seed}" >> $RESULTS_FILE
                else
                    echo "[!]    ${RESULTADO_VAL} - ${RESULTADO_ALL} : C=${c}, kernel=${kernel}, seed=${seed}" >> $RESULTS_FILE
                fi
            else
                echo "       C=${c}, kernel=${kernel}, seed=${seed} : ${RESULTADO_VAL} - ${RESULTADO_ALL}" >> $RESULTS_FILE
            fi
        done
        echo ""
    done
done