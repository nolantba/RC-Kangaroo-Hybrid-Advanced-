# RCKangaroo Hybrid+SOTA+ Makefile
# Combines GPU+CPU execution with SOTA+ optimizations
# Usage:
#   make clean
#   make SM=86 USE_JACOBIAN=1 PROFILE=release -j
#   ./rckangaroo -cpu 64 -dp 16 -range 84 ...

TARGET := rckangaroo

# Toolchains
CC    := g++
NVCC  := nvcc

# CUDA
CUDA_PATH ?= /usr
SM        ?= 86
USE_JACOBIAN ?= 1
USE_PERSISTENT_KERNELS ?= 0
USE_SOTA_PLUS ?= 1
USE_SEPARATE_HERD_KERNEL ?= 0
PROFILE   ?= release

# Separate optimization: host vs device
# Host optimizations: aggressive inlining, native arch, link-time optimization, AVX2
# AVX2 flags: -mavx2 -mfma for Xeon E5-2696 v3 (Haswell) and newer
HOST_COPT_release := -O3 -DNDEBUG -ffunction-sections -fdata-sections -march=native -mtune=native -mavx2 -mfma -flto -finline-functions -funroll-loops -fopenmp
HOST_COPT_debug   := -O0 -g -mavx2 -mfma
HOST_COPT := $(HOST_COPT_$(PROFILE))

DEV_COPT_release := -O3
DEV_COPT_debug   := -O0 -g
DEV_COPT := $(DEV_COPT_$(PROFILE))

# Flags
CCFLAGS    := -std=c++17 -I$(CUDA_PATH)/include $(HOST_COPT) -DUSE_JACOBIAN=$(USE_JACOBIAN) -DUSE_PERSISTENT_KERNELS=$(USE_PERSISTENT_KERNELS) -DUSE_SOTA_PLUS=$(USE_SOTA_PLUS) -DUSE_SEPARATE_HERD_KERNEL=$(USE_SEPARATE_HERD_KERNEL)
# Base NVCC flags (common to all SM architectures)
NVCCFLAGS_COMMON := -std=c++17 $(DEV_COPT) -Xptxas -O3 -Xfatbin=-compress-all -DUSE_JACOBIAN=$(USE_JACOBIAN) -DUSE_PERSISTENT_KERNELS=$(USE_PERSISTENT_KERNELS) -DUSE_SOTA_PLUS=$(USE_SOTA_PLUS) -DUSE_SEPARATE_HERD_KERNEL=$(USE_SEPARATE_HERD_KERNEL) --use_fast_math -allow-unsupported-compiler -ccbin g++-11

# SM-specific optimizations
ifeq ($(SM),80)
    # SM 8.0 (A100/RTX 3090) - Conservative flags for stability
    # Disable device LTO on SM 80 (can cause illegal memory access)
    NVCCFLAGS := $(NVCCFLAGS_COMMON) -arch=sm_$(SM) --extra-device-vectorization -rdc=true
else ifeq ($(SM),86)
    # SM 8.6 (RTX 3060/3070) - Aggressive cache control + LTO
    NVCCFLAGS := $(NVCCFLAGS_COMMON) -arch=sm_$(SM) -Xptxas -dlcm=ca -Xptxas --def-load-cache=ca -Xptxas --def-store-cache=wb -Xptxas=-allow-expensive-optimizations=true --extra-device-vectorization -dlto -rdc=true
else
    # Other architectures - Safe defaults
    NVCCFLAGS := $(NVCCFLAGS_COMMON) -arch=sm_$(SM) --extra-device-vectorization -rdc=true
endif
NVCCXCOMP  := -Xcompiler -ffunction-sections -Xcompiler -fdata-sections

LDFLAGS   := -L$(CUDA_PATH)/lib/x86_64-linux-gnu -lcudart -pthread -lcudadevrt
# Optional: Enable NVML for GPU monitoring (requires nvidia-ml-dev package)
ifdef USE_NVML
    CCFLAGS  += -DUSE_NVML
    LDFLAGS  += -lnvidia-ml
endif

# Sources (including CPU worker, save/resume system, GPU monitoring, and SOTA++ herds)
SRC_CPP := RCKangaroo.cpp GpuKang.cpp CpuKang.cpp CpuKang_JLP.cpp Ec.cpp Lambda.cpp utils.cpp WorkFile.cpp XorFilter.cpp GpuMonitor.cpp GpuHerdManager.cpp

# CUDA sources (including herd kernels)
CU_DIR ?= .
SRC_CU := $(wildcard $(CU_DIR)/RCGpuCore.cu) $(wildcard $(CU_DIR)/GpuHerdKernels.cu)

OBJ_CPP := $(SRC_CPP:.cpp=.o)
OBJ_CU  := $(patsubst %.cu,%.o,$(SRC_CU))

ifeq ($(strip $(OBJ_CU)),)
  $(warning [Makefile] No RCGpuCore.cu found in $(CU_DIR). Building CPU-only.)
  OBJS := $(OBJ_CPP)
else
  OBJS := $(OBJ_CPP) $(OBJ_CU)
endif

.PHONY: all clean print-vars

all: $(TARGET)

$(TARGET): $(OBJS)
	@# Device link step for CUDA
	@# Note: Device LTO disabled due to compatibility issues with CUDA 12.0
	$(NVCC) -arch=sm_$(SM) -dlink -o gpu_dlink.o $(OBJ_CU) -L$(CUDA_PATH)/lib/x86_64-linux-gnu -lcudadevrt
	@# Final host link with device-linked object
	$(CC) $(CCFLAGS) -o $@ $(OBJS) gpu_dlink.o $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CCFLAGS) -c $< -o $@

# Generic CUDA rule (.cu -> .o) with host flags via -Xcompiler
$(CU_DIR)/%.o: $(CU_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) $(NVCCXCOMP) -c $< -o $@

# Explicit rule
$(CU_DIR)/RCGpuCore.o: $(CU_DIR)/RCGpuCore.cu RCGpuUtils.h Ec.h defs.h
	$(NVCC) $(NVCCFLAGS) $(NVCCXCOMP) -c $< -o $@

# Herd kernels compilation
$(CU_DIR)/GpuHerdKernels.o: $(CU_DIR)/GpuHerdKernels.cu GpuHerdManager.h HerdConfig.h RCGpuUtils.h defs.h
	$(NVCC) $(NVCCFLAGS) $(NVCCXCOMP) -c $< -o $@

# Herd manager compilation
GpuHerdManager.o: GpuHerdManager.cpp GpuHerdManager.h HerdConfig.h defs.h utils.h
	$(CC) $(CCFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ_CPP) $(OBJ_CU) gpu_dlink.o $(TARGET)

print-vars:
	@echo "CUDA_PATH=$(CUDA_PATH)"
	@echo "SM=$(SM)"
	@echo "USE_JACOBIAN=$(USE_JACOBIAN)"
	@echo "PROFILE=$(PROFILE)"
	@echo "SRC_CPP=$(SRC_CPP)"
	@echo "CU_DIR=$(CU_DIR)"
	@echo "SRC_CU=$(SRC_CU)"
	@echo "OBJ_CPP=$(OBJ_CPP)"
	@echo "OBJ_CU=$(OBJ_CU)"
	@echo "OBJS=$(OBJS)"
	@echo "NVCCFLAGS=$(NVCCFLAGS)"
	@echo "NVCCXCOMP=$(NVCCXCOMP)"
