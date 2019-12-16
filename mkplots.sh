#!/bin/bash
FILES=plots/*.npy
for f in $FILES
do
  echo "Processing $f file..."
  python plot.py $f
done
