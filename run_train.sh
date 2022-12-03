#!/usr/bin/env bash

for nr in `echo 5 6 7 8`
do
        echo "nr:" ${nr}
        python train.py --nr ${nr} > log/log_simsiam_${nr}r_2048f_200e.txt
done