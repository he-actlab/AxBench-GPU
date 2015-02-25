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


echo -e "${Green} CUDA Image Segmentation Starting... ${White}"

for f in test.data/input/*.pnm
do
	filename=$(basename "$f")
	extension="${filename##*.}"
	filename="${filename%.*}"
	echo -e "-------------------------------------------------------"
	echo -e "${Green} Input Image:  $f"
	echo -e "${Green} output Image: ./train.data/output/${filename}_segmented.pgm ${White}"
	echo -e "-------------------------------------------------------"
	./bin/quickshift_nn.out --file=$f --mode=gpu --outfile=./test.data/output/${filename}_segmented_nn.pnm
	./bin/quickshift.out    --file=$f --mode=gpu --outfile=./test.data/output/${filename}_segmented.pnm
	
	compare -metric RMSE ./test.data/output/${filename}_segmented_nn.pnm ./test.data/output/${filename}_segmented.pnm null > tmp.log 2> tmp.err
	echo -ne "${red}$f\t"
	awk '{ printf("*** Error: %0.2f%\n",substr($2, 2, length($2) - 2) * 100) }' tmp.err
	echo -ne "${white}"
done
