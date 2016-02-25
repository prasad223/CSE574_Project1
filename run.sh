#!/bin/bash

hc=0
script_name="nnScript.py"
file_pre="out_"
for((hc=0;hc<=20;hc++))
do
  for((l=0;l<10;l++))
  do
    lmb='0.'$l
    out_file=$file_pre$hc"_"$lmb".out"
    echo `time python $script_name $hc $lmb`
  done
done


