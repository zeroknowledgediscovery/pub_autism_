#!/bin/bash

FILE=$1

pdfjam $FILE 2 -o specific_aims.pdf
pdfjam $FILE 3-14 -o strategy.pdf
pdfjam $FILE 15- -o reference.pdf

