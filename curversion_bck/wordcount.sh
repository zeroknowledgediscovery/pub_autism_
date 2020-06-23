#!/bin/bash

 texcount $1 | grep '+' | awk -F+ '{print $1}' | awk 'BEGIN{s=0}{s=s+$1}END{print s}'

