include ../config.mk
INCLUDES  := -I../../common/inc

CFLAGS		:= 
LFLAGS		:= -lParrot
HEADERS     := src
INCLUDE 	:= -I${ANALOG_INC} -I${FANN_INC} -I${HEADERS} -I/usr/local/cuda/include
LIB			:= -L${ANALOG_LIB} -L${FANN_LIB} -L$(PARROT_LIB) -L/usr/local/cuda/lib
MODULE		:= inversek2j.out 
LIB_CUDA	:= -lcuda -lcudart
#DEBUG       := -g -G -O0
################################################################################

# Target rules
all: DIR build

DIR:
	@echo ${CPP_FILES}
	@echo ${OBJ_FILES}
	if [ ! -d "./bin" ];then 	\
		mkdir bin;				\
	fi
	if [ ! -d "./obj" ];then 	\
		mkdir obj;				\
	fi


build: $(MODULE)

./obj/inversek2j.o:./src/main.cu
	python $(PLANG) -c $(PARROT_JSON) -a observe -e "nvcc $(CFLAGS) $(INCLUDE) -I$(NVIDIA_HELPER) $(INCLUDES) $(ALL_CCFLAGS) -gencode arch=compute_20,code=compute_20 $(LIB_CUDA) -I$(PARROT_LIB) -c" -s "$<" -o "$@" -t
	#nvcc -O3 $(DEBUG) $(INCLUDE) $(LIB) -o $@ -c $<

$(MODULE): ./obj/inversek2j.o
	nvcc -O3 $(DEBUG) -g -G $(INCLUDE) $(LIB) -o ./bin/$@ $+

clean:
	rm -f ./bin/$(MODULE)
	rm -f ./obj/inversek2j.o

clobber: clean
