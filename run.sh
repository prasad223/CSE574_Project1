#!/bin/bash

script_name="nnScript.py"
for((hc=4;hc<=20;hc+=4))
do
  for((l=0;l<10;l++))
  do
    lmb='0.'$l
    echo `time python $script_name $hc $lmb`
  done
done
echo `time python $script_name 20 1.0`

