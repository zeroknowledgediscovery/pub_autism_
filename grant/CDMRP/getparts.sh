#!/bin/bash

FILE=$1

pdfjam $FILE 1-2 --letterpaper -o narrative.pdf
pdfjam $FILE 3 --letterpaper -o abbrv.pdf
pdfjam $FILE 4- --letterpaper -o references.pdf
