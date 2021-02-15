#!/bin/bash

FILE=$1

pdfjam $FILE 2 --letterpaper -o specific_aims.pdf
pdfjam $FILE 3-8  --letterpaper -o strategy.pdf
pdfjam $FILE 9-  --letterpaper -o reference.pdf

