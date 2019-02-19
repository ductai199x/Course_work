#!/bin/bash

# Assumes the script is located in the data directory
for filename in ./*; do
    name=$(basename $filename .txt)
    echo "$filename -> $name"
    sed -i "s/rs1/rs1/g" $filename
    sed -i "s/rs2/rs2/g" $filename
done
