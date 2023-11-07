#!/bin/bash
#preprocess
preprocess_args_name=([1]='dataset' [2]='numOfTrainClients' [3]='numOfTestClients' [4]='seed'  [5]='numOfClasses')
preprocess_args_value=([1]='mnist' [2]='50' [3]='50' [4]='0' [5]='10')
preprocess_suffix=""
generate_preprocess_suffix(){
  suffix=""
  for index in ${!preprocess_args_name[@]};
  do
    suffix="$suffix--${preprocess_args_name[${index}]} ${preprocess_args_value[${index}]} ";
  done
  preprocess_suffix=$suffix
}
preprocess_args_value[1]='cifar'
generate_preprocess_suffix
python3 preprocess.py $preprocess_suffix 

preprocess_args_value[1]='mnist'
generate_preprocess_suffix
python3 preprocess.py $preprocess_suffix 

#train
public_args_name=( [1]='global_epochs' [2]='dataset' [3]='valset_ratio' [4]='seed' [5]='gpu' [6]='eval_while_training' [7]='log' [8]='eta'  [9]='experiment_number')
public_args_value=([1]='200'           [2]='mnist'   [3]='0.1'            [4]='17'   [5]='1'   [6]='1'                   [7]='0'   [8]='0.001'  [9]='4')

FedAvg_args_name=( [1]='algorithm' [2]='local_epochs')
FedAvg_args_value=([1]='FedAvg'    [2]='10'         )

public_suffix=""
FedAvg_suffix=""
generate_public_suffix(){
  suffix=""
  for index in ${!public_args_name[@]};
  do
    suffix="$suffix--${public_args_name[${index}]} ${public_args_value[${index}]} ";
  done
  public_suffix=$suffix
}
generate_FedAvg_suffix(){
  suffix=""
  for index in ${!FedAvg_args_name[@]};
  do
    suffix="$suffix--${FedAvg_args_name[${index}]} ${FedAvg_args_value[${index}]} ";
  done
  FedAvg_suffix=$suffix
}

#  1
public_args_value[2]="mnist" #set dataset to mnist
generate_public_suffix
generate_FedAvg_suffix
python3 main.py $public_suffix $FedAvg_suffix

#  2
public_args_value[2]="cifar" #set dataset to cifar
generate_public_suffix
generate_FedAvg_suffix
python3 main.py $public_suffix $FedAvg_suffix

