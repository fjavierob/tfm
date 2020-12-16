#!/bin/bash

ENV_="tfm36"
PYTHON_FILE="clasificador_knn.py"
RESULTS_TMP="./knn_tmp"
HIGH_ACC=70

# DATASET_DIR='../caracteristicas/'
# DATASET_NAME='dataset2_m1_3_7.csv'
# DATASET_FILE="${DATASET_DIR}""${DATASET_NAME}"

DATASET_NAME=$(echo ${DATASET_FILE##*/} | cut -f1 -d.)

# SAVE_BASE="./resultados/"
SAVE_DIR="${SAVE_BASE}knn/${DATASET_NAME}"
RESULTS_FILE="${SAVE_DIR}/resultados.txt"

# Parametros KNN
KMIN=1
KMAX=30
KSTEP=1
SEED_MIN=0
SEED_MAX=15

K=$(eval echo "{$KMIN..$KMAX..$KSTEP}")
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

for k in $K; do
    echo "K = $k"
    echo -n "  Seed: "
    for seed in $SEED; do
        echo -n "$seed "
        # Ejecutar KNN
        python $PYTHON_FILE $DATASET_FILE $seed $k $RESULTS_TMP >> /dev/null 2>&1
        # Resultado sin escalar los datos
        RESULTADO_VAL=$(cat $RESULTS_TMP | cut -f1 -d,)
        RESULTADO_ALL=$(cat $RESULTS_TMP | cut -f2 -d,) 
        RESULTADO_VAL_FLOOR="${RESULTADO_VAL:0:2}"
        RESULTADO_ALL_FLOOR="${RESULTADO_ALL:0:2}"
        if [[ RESULTADO_VAL_FLOOR -ge HIGH_ACC ]]; then
            if [[ RESULTADO_ALL_FLOOR -ge HIGH_ACC ]]; then
                echo "[!!!]  ${RESULTADO_VAL} - ${RESULTADO_ALL} : K=${k}, seed=${seed}" >> $RESULTS_FILE
            else
                echo "[!]    ${RESULTADO_VAL} - ${RESULTADO_ALL} : K=${k}, seed=${seed}" >> $RESULTS_FILE
            fi
        else
            echo "       K=${k}, seed=${seed}    : ${RESULTADO_VAL} - ${RESULTADO_ALL}" >> $RESULTS_FILE
        fi
        # Resultado con datos escalados
        RESULTADO_VAL=$(cat $RESULTS_TMP | cut -f3 -d,)
        RESULTADO_ALL=$(cat $RESULTS_TMP | cut -f4 -d,) 
        RESULTADO_VAL_FLOOR="${RESULTADO_VAL:0:2}"
        RESULTADO_ALL_FLOOR="${RESULTADO_ALL:0:2}"
        if [[ RESULTADO_VAL_FLOOR -ge HIGH_ACC ]]; then
            if [[ RESULTADO_ALL_FLOOR -ge HIGH_ACC ]]; then
                echo "[!!!]  ${RESULTADO_VAL} - ${RESULTADO_ALL} : K=${k}, seed=${seed}, S" >> $RESULTS_FILE
            else
                echo "[!]    ${RESULTADO_VAL} - ${RESULTADO_ALL} : K=${k}, seed=${seed}, S" >> $RESULTS_FILE
            fi
        else
            echo "       K=${k}, seed=${seed}, S : ${RESULTADO_VAL} - ${RESULTADO_ALL}" >> $RESULTS_FILE
        fi
    done
        echo ""
done 
