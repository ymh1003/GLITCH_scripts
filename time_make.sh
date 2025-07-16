#!/bin/bash

set -e

LF=log-`date +%Y%m%d-%H%M%S`.log
echo "*** TIME: " "make $@" >> $LF
make GLITCH_RUNNER="/usr/bin/time -p glitch_runner.py" GCODE_COMP_Z="/usr/bin/time -p gcode_comp_Z.py" "$@" 2>&1 | tee -a $LF
echo "*** SUCCESS"

