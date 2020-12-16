#!/bin/bash

ENV_="tfm36"
PYTHON_FILE="clasificador_rf.py"
RESULTS_TMP="./rf_tmp"
HIGH_ACC=73

# DATASET_DIR='../caracteristicas/'
# DATASET_NAME='dataset2_m1_3_7.csv'
# DATASET_FILE="${DATASET_DIR}""${DATASET_NAME}"

DATASET_NAME=$(echo ${DATASET_FILE##*/} | cut -f1 -d.)

# SAVE_BASE="./resultados/"
SAVE_DIR="${SAVE_BASE}rf/${DATASET_NAME}"
RESULTS_FILE="${SAVE_DIR}/resultados.txt"
# Parametros RF
DEPTH_MIN=1
# DEPTH_MAX=50
DSTEP=1
SEED_MIN=0
SEED_MAX=20

DEPTH=$(eval echo "{$DEPTH_MIN..$DEPTH_MAX..$DSTEP}")
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

for depth in $DEPTH; do
    echo "MAX_DEPTH = $depth"
    echo -n "  Seed: "
    for seed in $SEED; do
        echo -n "$seed "
        # Ejecutar RF
        python $PYTHON_FILE $DATASET_FILE $seed $depth $RESULTS_TMP >> /dev/null 2>&1
        # Resultado sin escalar los datos
        RESULTADO_VAL=$(cat $RESULTS_TMP | cut -f1 -d,)
        RESULTADO_ALL=$(cat $RESULTS_TMP | cut -f2 -d,) 
        RESULTADO_VAL_FLOOR="${RESULTADO_VAL:0:2}"
        RESULTADO_ALL_FLOOR="${RESULTADO_ALL:0:2}"
        if [[ RESULTADO_VAL_FLOOR -ge HIGH_ACC ]]; then
            if [[ RESULTADO_ALL_FLOOR -ge HIGH_ACC ]]; then
                echo "[!!!]  ${RESULTADO_VAL} - ${RESULTADO_ALL} : MAX_DEPTH=${depth}, seed=${seed}" >> $RESULTS_FILE
            else
                echo "[!]    ${RESULTADO_VAL} - ${RESULTADO_ALL} : MAX_DEPTH=${depth}, seed=${seed}" >> $RESULTS_FILE
            fi
        else
            echo "       MAX_DEPTH=${depth}, seed=${seed}    : ${RESULTADO_VAL} - ${RESULTADO_ALL}" >> $RESULTS_FILE
        fi
        # Resultado con PCA
        RESULTADO_VAL=$(cat $RESULTS_TMP | cut -f3 -d,)
        RESULTADO_ALL=$(cat $RESULTS_TMP | cut -f4 -d,) 
        RESULTADO_VAL_FLOOR="${RESULTADO_VAL:0:2}"
        RESULTADO_ALL_FLOOR="${RESULTADO_ALL:0:2}"
        if [[ RESULTADO_VAL_FLOOR -ge HIGH_ACC ]]; then
            if [[ RESULTADO_ALL_FLOOR -ge HIGH_ACC ]]; then
                echo "[!!!]  ${RESULTADO_VAL} - ${RESULTADO_ALL} : MAX_DEPTH=${depth}, seed=${seed}, P" >> $RESULTS_FILE
            else
                echo "[!]    ${RESULTADO_VAL} - ${RESULTADO_ALL} : MAX_DEPTH=${depth}, seed=${seed}, P" >> $RESULTS_FILE
            fi
        else
            echo "       MAX_DEPTH=${depth}, seed=${seed}, S : ${RESULTADO_VAL} - ${RESULTADO_ALL}" >> $RESULTS_FILE
        fi
    done
        echo ""
done 
