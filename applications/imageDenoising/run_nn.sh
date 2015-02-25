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

application=imageDenoising

echo -e "${green} Image Denoising Starting... ${white}"


for f in test.data/input/*.bmp
do
	filename=$(basename "$f")
	extension="${filename##*.}"
	filename="${filename%.*}"
	echo -e "-------------------------------------------------------"
	echo -e "${green} Input Image:  $f"
	echo -e "${green} output Image: ./test.data/output/${filename}_imageDenoised.ppm ${white}"
	echo -e "-------------------------------------------------------"
	./bin/${application}_nn.out $f ./test.data/output/${filename}_${application}_nn.ppm 1
	./bin/${application}.out    $f ./test.data/output/${filename}_${application}.ppm 1
	compare -metric RMSE ./test.data/output/${filename}_${application}_nn.ppm ./test.data/output/${filename}_${application}.ppm null > tmp.log 2> tmp.err
	echo -ne "${red}$f\t"
	awk '{ printf("*** Error: %0.2f%\n",substr($2, 2, length($2) - 2) * 100) }' tmp.err
	echo -ne "${white}"
done
