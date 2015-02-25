#!/bin/bash

# Regular Colors
Black='\e[0;30m'        # Black
Red='\e[0;31m'          # Red
Green='\e[0;32m'        # Green
Yellow='\e[0;33m'       # Yellow
Blue='\e[0;34m'         # Blue
Purple='\e[0;35m'       # Purple
Cyan='\e[0;36m'         # Cyan
White='\e[0;37m'        # White


echo -e "${Green} CUDA FFT2D Starting... ${White}"

if [ ! -d ./train.data/output/kernel.data ]; then
	mkdir ./train.data/output/kernel.data
fi


./bin/convolutionFFT2D_nn.out
mv result.data ./test.data/output/result_nn.data

./bin/convolutionFFT2D.out
mv result.data ./test.data/output/result_orig.data

python ./scripts/qos.py ./test.data/output/result_orig.data ./test.data/output/result_nn.data

