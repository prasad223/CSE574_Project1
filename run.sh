#!/bin/bash

script_name="nnScript.py"
lamb=$(awk 'BEGIN{for(i=0;i<=1;i+=0.1)print i}')
for((hc=4;hc<=20;hc+=4))
do
  for l in $lamb
    do
      echo $script_name $hc $l
      echo `time python3 $script_name $hc $l`
    done
done


