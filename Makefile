# Binary file
BIN = main_yielding_athermal_creep

# Pathts & Includes
CUDA_INSTALL_PATH	= /usr/local/cuda
CUDA_LIB_PATH		= $(CUDA_INSTALL_PATH)/lib64
INCLUDES		+= -I. -I/home/ferrero/SimpleYielding/common/ 

# Flags
LIBS			= #-L$(CUDA_LIB_PATH)
CFLAGS			= -O3 #TODO:try -06
#CFLAGS			= -O -g
LDFLAGS			= -lcufft #-lrt -lpthread
NVCC_FLAGS		= -O3 --use_fast_math #-arch=sm_35  --ptxas-options=-v #-ftz=true 

# Default Values
LX			= 256
LY			= 256
TIME_STEP		= 0.1
FF			= 1.0
EE			= 1.0
#CUDA_DEVICE		= 0
OFILE			= out_LX$(LX)LY$(LY)_dt$(TIME_STEP).dat

# Simulation Parameters & Compilation Options
PARAMETERS		= -DLX=$(LX) -DLY=$(LY) -DTIME_STEP=$(TIME_STEP) ##-DFF=$(FF) -DEE=$(EE)
OPTIONS			= -DFIX_EXT_STRESS -DEQUILIBRATION -DDOUBLE_PRECISION -DPARABOLAS #-DSINUSOIDAL -DJUSTFLOWCURVE #-DFROMBELOW #-DFIX_EXT_STRESS -DFIX_EXT_STRESS ##-DEQUILIBRATION #-DPRINTSTRAINJUMPS #-DPRINTSTRAIN #-DOPT -DPRINTFORCES -DTIMEFIX  #-DVISUALIZATION -DJPEG 

# Compilers
NVCC                    = nvcc #$(CUDA_INSTALL_PATH)/bin/nvcc
#VCC                    = nvcc
GCC                     = g++
LINKER                  = nvcc #$(CUDA_INSTALL_PATH)/bin/nvcc

# Files
CPP_SOURCES             = mt.cpp
CU_SOURCES              = $(BIN).cu
HEADERS                 =
CPP_OBJS                = $(patsubst %.cpp, %.o, $(CPP_SOURCES))
CU_OBJS                 = $(patsubst %.cu, %.o, $(CU_SOURCES))

# Rules
$(BIN): clean $(CPP_OBJS) $(CU_OBJS) $(HEADERS)
	$(LINKER) -o $(BIN) $(CU_OBJS) $(CPP_OBJS) $(LDFLAGS) $(INCLUDES) $(LIBS)

$(CPP_OBJS): $(CPP_SOURCES) $(HEADERS)
	$(GCC) -c $(CPP_SOURCES) $(CFLAGS) $(INCLUDES)

$(CU_OBJS): $(CU_SOURCES) $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) -c $(CU_SOURCES) $(PARAMETERS) $(OPTIONS) $(INCLUDES)

run: $(BIN)
	LD_LIBRARY_PATH=$(CUDA_LIB_PATH) ./$(BIN) > $(OFILE) &

clean:
	rm -f $(BIN) *.o *.linkinfo cuda_profile_?.log

