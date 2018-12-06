# Location of the CUDA Toolkit
CUDA_PATH ?= "/usr/local/cuda"

TARGET_SIZE :=64

HOST_COMPILER := g++
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

NVCCFLAGS   := -m${TARGET_SIZE} --std=c++14
NVCCDEBUGFLAGS := 
NVCCOPTFLAGS :=
CFLAGS     := -fpermissive -fopenmp -Winline
LDFLAGS     := -lgomp
CDEFINES :=

CCFLAGS += $(addprefix -Xcompiler ,$(CFLAGS))
LLDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))

INCLUDES += -I/opt/packages/opencv/3.1.0/include
LIBRARIES := -L/opt/packages/opencv/3.1.0/lib -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_core -lopencv_videostab -lopencv_videoio -lopencv_imgcodecs -lopencv_video -lnppisu -lnppist -lnppicc 

################################################################################

# Gencode arguments
SMS ?= 50

ifeq ($(GENCODE_FLAGS),)
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))
# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
endif

################################################################################
TARGETNAME:= figmp_test

# Target rules
all: release

release: NVCCDEBUGFLAGS = 
release: NVCCOPTFLAGS = -O3 --use_fast_math 
release: $(TARGETNAME)

debug: NVCCDEBUGFLAGS = -g -G
debug: CDEFINES_AUX= -DCUDA_DEBUG
debug: CDEFINES=$(addprefix -Xcompiler ,$(CDEFINES_AUX))
debug: $(TARGETNAME)

current_test.o: current_test.cu
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) $(NVCCOPTFLAGS) $(NVCCDEBUGFLAGS) $(CCFLAGS) $(CDEFINES) $(GENCODE_FLAGS) -o $@ -c $<

$(TARGETNAME): current_test.o
	$(NVCC)  $(NVCCFLAGS) $(NVCCOPTFLAGS) $(NVCCDEBUGFLAGS) $(CCFLAGS) $(CDEFINES) $(LLDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

clean:
	rm -f $(TARGETNAME) *.o
