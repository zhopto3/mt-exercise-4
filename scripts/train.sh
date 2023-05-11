#! /bin/bash
#Zachary William Hopton, 22-737-027
#Allison Ponce de Leon Diaz 22-740-633
scripts=$(dirname "$0")
base=$scripts/..

models=$base/models
configs=$base/configs

mkdir -p $models

#changed for the number of CPU cores we have.
num_threads=2
#device=0

# measure time
#The below is for to train the default model provided
#SECONDS=0
#
#logs=$base/logs
#
#model_name=deen_transformer_regular
#
#mkdir -p $logs
#
#mkdir -p $logs/$model_name
#
#OMP_NUM_THREADS=$num_threads python -m joeynmt train $configs/$model_name.yaml > $logs/$model_name/out 2> $logs/$model_name/err
#
#echo "time taken:"
#echo "$SECONDS seconds"

echo "Now starting the pre-layer normalization" model training

SECONDS=0

logs=$base/logs

model_name=deen_transformer_pre

mkdir -p $logs

mkdir -p $logs/$model_name

OMP_NUM_THREADS=$num_threads python -m joeynmt train $configs/$model_name.yaml > $logs/$model_name/out 2> $logs/$model_name/err

echo "time taken:"
echo "$SECONDS seconds"

echo "Now starting the post-layer normalization" model training

SECONDS=0

logs=$base/logs

model_name=deen_transformer_post

mkdir -p $logs

mkdir -p $logs/$model_name

OMP_NUM_THREADS=$num_threads python -m joeynmt train $configs/$model_name.yaml > $logs/$model_name/out 2> $logs/$model_name/err

echo "time taken:"
echo "$SECONDS seconds"