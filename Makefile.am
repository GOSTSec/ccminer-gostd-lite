# allow to use Host cuda functions in C/C++
DEF_INCLUDES = @CUDA_INCLUDES@

if WANT_JANSSON
JANSSON_INCLUDES= -I$(top_srcdir)/compat/jansson
else
JANSSON_INCLUDES=
endif

EXTRA_DIST =
	autogen.sh README.md README.txt LICENSE.txt \
	ccminer.sln ccminer.vcxproj ccminer.vcxproj.filters \
	compat/gettimeofday.c compat/getopt/getopt_long.c

SUBDIRS = compat

bin_PROGRAMS = ccminer

ccminer_SOURCES = 
	api.cpp bench.cpp bignum.cpp ccminer.cpp compat.h crc32.c cuda.cpp hashlog.cpp nvml.cpp nvsettings.cpp \
	pools.cpp stats.cpp sysinfos.cpp util.cpp \
	compat/getopt/getopt.h compat/inttypes.h compat/stdbool.h compat/sys/time.h compat/unistd.h \
	elist.h miner.h uint256.h cuda_checkhash.cu sph/streebog.c gost/cuda_gosthash.cu gost/gost.cu \

if HAVE_NVML
nvml_defs = -DUSE_WRAPNVML
nvml_libs = -ldl
endif

if HAVE_WINDOWS
ccminer_SOURCES += compat/winansi.c
endif

ccminer_LDFLAGS  = $(PTHREAD_FLAGS) @CUDA_LDFLAGS@
ccminer_LDADD    = @LIBCURL@ @JANSSON_LIBS@ @PTHREAD_LIBS@ @WS2_LIBS@ @CUDA_LIBS@ @OPENMP_CFLAGS@ @LIBS@ $(nvml_libs)
ccminer_CPPFLAGS = @LIBCURL_CPPFLAGS@ @OPENMP_CFLAGS@ $(CPPFLAGS) $(PTHREAD_FLAGS) -fno-strict-aliasing $(JANSSON_INCLUDES) $(DEF_INCLUDES) $(nvml_defs)

if HAVE_OSX
ccminer_CPPFLAGS += -I/usr/local/llvm/lib/clang/4.0.0/include
ccminer_LDFLAGS += -L/usr/local/llvm/lib
ccminer_LDADD += -lomp
endif

#ccminer_CPPFLAGS += -DUSE_LIBSODIUM
#ccminer_LDFLAGS += -Lequi/lib
#ccminer_LDADD += -lsodium
ccminer_LDADD += -lcuda

nvcc_ARCH  = -gencode=arch=compute_50,code=\"sm_50,compute_50\"

#nvcc_ARCH += -gencode=arch=compute_52,code=\"sm_52,compute_52\"
#nvcc_ARCH += -gencode=arch=compute_35,code=\"sm_35,compute_35\"
#nvcc_ARCH += -gencode=arch=compute_30,code=\"sm_30,compute_30\"
#nvcc_ARCH += -gencode=arch=compute_20,code=\"sm_21,compute_20\"

nvcc_FLAGS = $(nvcc_ARCH) @CUDA_INCLUDES@ -I. @CUDA_CFLAGS@
nvcc_FLAGS += $(JANSSON_INCLUDES) --ptxas-options="-v"

# we're now targeting all major compute architectures within one binary.
.cu.o:
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=128 -o $@ -c $<
