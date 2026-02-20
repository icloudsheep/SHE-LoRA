#!/bin/bash

# conda init
# if ! command -v conda &>/dev/null; then
#     echo "Conda is not installed or not in PATH."
#     exit 1
# fi


# conda activate she-lora


# if [ $? -ne 0 ]; then
#     echo "Failed to activate Conda environment 'she-lora'."
#     exit 1
# fi


# export PYDEVD_GDB_SCAN_SHARED_LIBRARIES="libdl,libc,libpthread,libm"
# export PYDEVD_DISABLE_FILE_VALIDATION="1"
# export OMP_NUM_THREADS="1"

python -Xfrozen_modules=off "${PWD}/pythonic_starter.py"