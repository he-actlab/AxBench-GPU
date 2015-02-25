#!/bin/bash

# Regular Colors
black='\e[0;30m'        # Black
red='\e[0;31m'          # Red
green='\e[0;32m'        # Green
yellow='\e[0;33m'       # Yellow
blue='\e[0;34m'         # Blue
purple='\e[0;35m'       # Purple
cyan='\e[0;36m'         # Cyan
white='\e[0;37m'        # White

application=dct8x8

echo -e "${Green} CUDA Sobel Edge-Detection Starting... ${White}"

if [ ! -d ./train.data/output/kernel.data ]; then
	mkdir ./train.data/output/kernel.data
fi

for f in test.data/input/*.bmp
do
	filename=$(basename "$f")
	extension="${filename##*.}"
	filename="${filename%.*}"
	echo -e "-------------------------------------------------------"
	echo -e "${Green} Input Image:  $f"
	echo -e "${Green} output Image: ./test.data/output/${filename}_dct_idct.bmp ${White}"
	echo -e "-------------------------------------------------------"
	./bin/${application}.out $f ./test.data/output/${filename}_dct_idct.bmp
	./bin/${application}_nn.out $f ./test.data/output/${filename}_dct_idct_nn.bmp
	compare -metric RMSE ./test.data/output/${filename}_dct_idct.bmp ./test.data/output/${filename}_dct_idct_nn.bmp null > tmp.log 2> tmp.err
	echo -ne "${red}$f\t"
	awk '{ printf("*** Error: %0.2f%\n",substr($2, 2, length($2) - 2) * 100) }' tmp.err
	echo -ne "${white}"
done
