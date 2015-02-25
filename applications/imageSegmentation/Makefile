Darwin_i386_ARCH := maci
Linux_i386_ARCH := glx
Linux_i686_ARCH := glx
Linux_x86_64_ARCH := a64

UNAME := $(shell uname -sm)
ARCH ?= $($(shell echo "$(UNAME)" | tr \  _)_ARCH)

CFLAGS   = -fno-strict-aliasing -O3 -DUNIX 

# Mac OS X Intel 32 / 64
ifeq ($(ARCH),maci)
GPUROOT := /usr/local/cuda
GPULIB  := $(GPUROOT)/lib
GDKROOT := /Developer/GPU\ Computing
SDKROOT := /Developer/SDKs/MacOSX10.5.sdk
MACOPTS := -m32 
MACLD   := $(MACOPTS) -Xlinker -rpath $(GPUROOT)/lib
LDFLAGS += -lm -mmacosx-version-min=10.5
CPU := i386
PLATFORM := darwin
endif


# Linux 32
ifeq ($(ARCH),glx)
GPUROOT := /usr/local/cuda
GPULIB  := $(GPUROOT)/lib
GDKROOT := /home/brian/NVIDIA_GPU_Computing_SDK
MACOPTS := 
MACLD   := 
CPU := i386
PLATFORM := linux
endif

# Linux 64
ifeq ($(ARCH),a64)
GPUROOT := /usr/local/cuda
GPULIB  := $(GPUROOT)/lib64
GDKROOT := /home/brian/NVIDIA_GPU_Computing_SDK
MACOPTS := 
MACLD   := 
CPU := x86_64
PLATFORM := linux
endif

include ../config.mk

CFLAGS += -I. -I$(GPUROOT)/include -I$(GDKROOT)/C/common/inc -I$(GDKROOT)/shared/inc

NVCC     = $(GPUROOT)/bin/nvcc
GENCODE  = -gencode=arch=compute_20,code=sm_20,compute_20  
LDFLAGS += -L$(GPULIB) -L$(GDKROOT)/C/lib -L$(GDKROOT)/C/common/lib/$(PLATFORM) -L$(GDKROOT)/shared/lib
LDFLAGS += -lcudart -lcutil_$(CPU) -lshrutil_$(CPU)
MODULE := quickshift.out

all: DIR $(MODULE)

DIR:
	@echo ${CPP_FILES}
	@echo ${OBJ_FILES}
	if [ ! -d "./bin" ];then 	\
		mkdir bin;				\
	fi
	if [ ! -d "./obj" ];then 	\
		mkdir obj;				\
	fi

./obj/%.o: ./src/%.cpp
	python $(PLANG) -c $(PARROT_JSON) -a observe -e "g++ $(MACOPTS) -Wall $(CFLAGS) -I$(CUTIL) -I./src -c" -s "$<" -o "$@" -t
	#g++ $(MACOPTS) -Wall $(CFLAGS) -I$(CUTIL) -o $@ -c $<

./obj/quickshift_gpu.o: ./src/quickshift_gpu.cu
	python $(PLANG) -c $(PARROT_JSON) -a observe -e "$(NVCC) -arch=compute_20 -code=sm_20,compute_20 -I$(CUTIL) -I./src $(MACOPTS) --compiler-options $(CFLAGS) -c" -s "$<" -o "$@" -t
	#$(NVCC) $(GENCODE) -I$(CUTIL) $(MACOPTS) --compiler-options $(CFLAGS) -o $@ -c $<

$(MODULE): ./obj/Image.o ./obj/quickshift_cpu.o ./obj/main.o ./obj/quickshift_gpu.o
	g++ -fPIC $(MACLD) -o ./bin/$(MODULE) $^ $(LDFLAGS)

clean:
	rm -f ./bin/$(MODULE)
	rm -f ./obj/Image.o ./obj/quickshift_cpu.o ./obj/main.o ./obj/quickshift_gpu.o
