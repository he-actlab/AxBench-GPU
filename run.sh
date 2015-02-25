#!/bin/bash

USAGE="usage: run.sh [setup] [application name] | [make|clean|run] [blackscholes|dct|fft|imageDenoising|convolution|segmentation|imageBinarization|jmeint|jpeg|kmeans|sobel|all]"


red='\033[0;31m'
blue='\033[0;34m'
green='\033[0;32m'
nc='\033[0m' # No Color
underline=`tput smul`
normal=`tput sgr0`

kernelArray=()


function printUsage()
{
	echo -en "\033[31m"
    echo $USAGE
    echo -en "\033[0m"
}

function printRun()
{
	echo -en "\033[36m"
    echo "running test suite ** $1 **"
    echo -en "\033[0m"
}

getArray() {
    i=0
    while read line # Read a line
    do
        kernelArray[i]=$line # Put it into the array
        i=$(($i + 1))
    done < $1
}




function MakeSrc()
{
	if [ ! -d applications/$1 ]
	then
		echo "$1 doesn't exist..."
		exit 
	fi	
	echo "enter $1, making..."
	cd applications/$1
	make &> ./log/Make.log
	if grep -q "error" ./log/Make.log; then
		grep -rn "error" ./log/Make.log
    	echo -e "${red}---------- application ** $1 ** failed during compiling (check ${applications}/log/Make.log) ----------${nc}"
    	exit 1
	fi
	echo -en "\033[36m"
	echo ""
	echo "---------- application ** $1 ** made successfully ----------"
	echo ""
	echo -en "\033[0m"
	cd -
}

function CleanSrc()
{
	if [ ! -d applications/$1 ]
	then
		echo "$1 doesn't exist..."
		exit 
	fi	
	echo "enter $1, cleaning..."
	cd applications/$1
	make clean &> ./log/clean.log
	if grep -q "error" ./log/clean.log; then
		grep -rn "error" ./log/clean.log
    	echo -e "${red}---------- application ** $1 ** failed during cleaning (check ${applications}/log/clean.log) ----------${nc}"
    	exit 1
	fi
	echo -en "\033[36m"
	echo ""
	echo "---------- application ** $1 ** has been successfully cleaned ----------"
	echo ""
	echo -en "\033[0m"
	cd -
}

function RunSrc()
{
	if [ ! -d applications/$1 ]
	then
		echo "$1 doesn't exist..."
		exit 
	fi		
	cd applications/$1
	echo -e "${green}#1: Collect the training data...${nc}"
	bash run_observation.sh
	echo -e "${green}#2: Aggregate the training data...${nc}"
	python ../../scripts/dataConv.py ./train.data/output/kernel.data
	python ../../scripts/readKernelNames.py
	getArray kernelNames.tmp
	for k in "${kernelArray[@]}"
	do
		echo -e "${green} >>>>>> Start Training for Kernel ${k} <<<<<<"
		echo -ne "${blue} Do you want to perform training for this kernel?[y/N] ${nc}"
		read kernelAns
		while [[ "${kernelAns}" != "y" && "${kernelAns}" != "N" ]]; do
			echo -ne "${blue} Do you want to perform training for this kernel?[y/N] ${nc}"
			read kernelAns
		done

		if [[ "${kernelAns}" == "N" ]]; then
			rm -rf fann.config/${k}.json
			rm -rf fann.config/${k}_cuda.txt
			rm -rf fann.config/${k}.nn
			rm -rf fann.config/activationFunc_${k}.cfg
			continue
		fi

		echo -e "${green}#3: Provide the compile parameters...${nc}"
		python ../../scripts/comm_to_json.py ${1}_${k}
		echo -e "${green}#4: Explore different NN topologies for each kernel...${nc}"
		python ../../scripts/train.py ${1}_${k} ${k} > ./log/${1}_training.log
		echo -e "${green}#5: Find the best NN topology...${nc}"
		python ../../scripts/find_best_NN.py $1 ${k}
	done
	# convert the best NN to cuda kernel
	for nn in ./fann.config/*.nn; do
		python ../../scripts/fann2kernel.py ${nn}
	done

	# end of training for each kernel
	echo -e "${green}#6: Replace the code with NN...${nc}"
	python ../../scripts/parrotConv.py $1
	echo -e "${green}#7: Compile the code with NN...${nc}"
	make -f Makefile_nn &> ./log/Make_nn.log
	if grep -q "error" ./log/Make_nn.log; then
		grep -rn "error" ./log/Make_nn.log
		echo -e "---------------------------------------------------------"
    	echo -e "${red}${underline} The transformed code for ** $1 ** failed during compiling (check ${applications}/log/Make_nn.log) ${normal}${nc}"
    	echo -e "---------------------------------------------------------"
    	exit 1
    else
    	echo -e "---------------------------------------------------------"
    	echo -e "${underline} The transformed code for ** $1 ** was successfully compiled ${normal}"
    	echo -e "---------------------------------------------------------"
	fi
	echo -e "${green}#8: Run the code on the test data...${nc}" # with Quality
	echo -e "---------------------------------------------------------"
	bash run_nn.sh
	echo -e "---------------------------------------------------------"
	echo ""
	echo -en "\033[0m"
	cd - > /dev/null
}

function SetupDir()
{
	if [ -d applications/$1 ]
		then
		echo -e "${red}Application ** $1 ** existed in the applications directory!${nc}"
		exit
	fi

	# base directory
	mkdir applications/$1
	# fann configs
	mkdir applications/$1/fann.config
	# log
	mkdir applications/$1/log
	# nn.configs
	mkdir applications/$1/nn.configs
	# src
	mkdir applications/$1/src
	# NN src
	mkdir applications/$1/src.nn
	# train data
	mkdir applications/$1/train.data
	# input and ouput
	mkdir applications/$1/train.data/input
	mkdir applications/$1/train.data/output
	mkdir applications/$1/train.data/output/fann.data
	mkdir applications/$1/train.data/output/bin
	# test data
	mkdir applications/$1/test.data
	# input and ouput
	mkdir applications/$1/test.data/input
	mkdir applications/$1/test.data/output

	echo -e "${green}Application ** $1 ** has created successfully!${nc}"	
}

#check the number of command line arguments
if [ $# -lt 2 ]
then
	printUsage
    exit
fi
if [ "$1" = "make" ]
then
	case $2 in
		"blackscholes")
			MakeSrc BlackScholes
		;;
		"dct")
			MakeSrc dct8x8
		;;
		"fft")
			MakeSrc convolutionFFT2D
		;;
		"inversek2j")
			MakeSrc $2
		;;
		"segmentation")
			MakeSrc imageSegmentation
		;;
		"imageDenoising")
			MakeSrc $2
		;;
		"imageBinarization")
			MakeSrc imageBinarization
		;;
		"convolution")
			MakeSrc convolutionSeparable
		;;
		"jmeint")
			MakeSrc $2
		;;
		"jpeg")
			MakeSrc $2
		;;
		"kmeans")
			MakeSrc $2
		;;
		"sobel")
			MakeSrc SobelFilter
		;;
		"all")
			MakeSrc BlackScholes
			MakeSrc fft
			MakeSrc inversek2j
			MakeSrc	jmeint
			MakeSrc jpeg
			MakeSrc kmeans
			MakeSrc sobel
		;;
	*)
		printUsage
		exit
		;;
	esac
elif [ "$1" = "clean" ]
then
	case $2 in
		"blackscholes")
			CleanSrc BlackScholes
		;;
		"dct")
			CleanSrc dct8x8
		;;
		"fft")
			CleanSrc convolutionFFT2D
		;;
		"imageDenoising")
			CleanSrc $2
		;;
		"segmentation")
			CleenSrc imageSegmentation
		;;
		"convolution")
			CleanSrc convolutionSeparable
		;;
		"imageBinarization")
			CleanSrc imageBinarization
		;;
		"inversek2j")
			CleanSrc $2
		;;
		"jmeint")
			CleanSrc $2
		;;
		"jpeg")
			CleanSrc $2
		;;
		"kmeans")
			CleanSrc $2
		;;
		"sobel")
			CleanSrc SobelFilter
		;;
		"all")
			CleanSrc BlackScholes
			CleanSrc fft
			CleanSrc inversek2j
			CleanSrc	jmeint
			CleanSrc jpeg
			CleanSrc kmeans
			CleanSrc sobel
		;;
	*)
		printUsage
		exit
		;;
	esac
elif [ "$1" = "run" ]
then
		case $2 in
		"blackscholes")
			RunSrc BlackScholes
		;;
		"dct")
			RunSrc dct8x8
		;;
		"fft")
			RunSrc convolutionFFT2D
		;;
		"imageDenoising")
			RunSrc $2
		;;
		"convolution")
			RunSrc convolutionSeparable
		;;
		"segmentation")
			RunSrc imageSegmentation
		;;
		"imageBinarization")
			RunSrc imageBinarization
		;;
		"inversek2j")
			RunSrc $2
		;;
		"jmeint")
			RunSrc $2
		;;
		"jpeg")
			RunSrc $2
		;;
		"kmeans")
			RunSrc $2
		;;
		"sobel")
			RunSrc SobelFilter
		;;
		"all")
			RunSrc BlackScholes
			RunSrc fft
			RunSrc inversek2j
			RunSrc jmeint
			RunSrc jpeg
			RunSrc kmeans
			RunSrc sobel
		;;
	*)
		printUsage
		exit
		;;
	esac
elif [ "$1" = "setup" ]
then
	SetupDir $2
else
	printUsage
	exit
fi