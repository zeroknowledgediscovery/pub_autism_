#!/bin/bash

FILE=$1

pdfjam $FILE 1 --letterpaper -o summary.pdf
pdfjam $FILE 2 --letterpaper -o narrative.pdf
pdfjam $FILE 3-6 --letterpaper -o facilities.pdf
pdfjam $FILE 8 --letterpaper -o specific_aims.pdf
pdfjam $FILE 9-14  --letterpaper -o strategy.pdf
pdfjam $FILE 15-  --letterpaper -o reference.pdf

