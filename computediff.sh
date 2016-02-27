#! /bin/sh
sed 's/Train_label: //' $1 | sed 's/predicted_label: //' | awk -f compare.awk
