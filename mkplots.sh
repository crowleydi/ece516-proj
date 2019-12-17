#!/bin/bash
FILES=keep/*.npy
for f in $FILES
do
  echo "Processing $f file..."
  python plot.py $f
done
