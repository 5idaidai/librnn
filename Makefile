# Code coverage advice
# export GCOV_PREFIX=testcov/
# export GCOV_PREFIX_STRIP=1
# http://nicolas.limare.net/pro/notes/2014/10/31_cblas_clapack_lapacke/
# http://stackoverflow.com/questions/137038/how-do-you-get-assembler-output-from-c-c-source-in-gcc -fverbose...

UNAME := $(shell uname)

ifeq ($(UNAME), Linux)
	CC := g++
	LIB := -lblas
endif
ifeq ($(UNAME), Darwin)
	CC  := g++
	LIB := -framework Accelerate
endif

SRCEXT      := cpp
SRCDIR      := src
BUILDDIR    := build
TEST_DIR    := test
TARGET      := bin/concurrent
TEST_TARGET := bin/test

SOURCES         := $(shell find $(SRCDIR) -type f -name *.$(SRCEXT))
TEST_SOURCES    := $(shell find $(TEST_DIR) -type f -name *.$(SRCEXT))

OBJECTS         := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.$(SRCEXT)=.o))
TEST_OBJECTS    := $(patsubst $(TEST_DIR)/%,$(BUILDDIR)/%,$(TEST_SOURCES:.$(SRCEXT)=.o))
OBJECTS_NO_MAIN := $(filter-out build/main.o, $(OBJECTS))

TEST_FLAGS  := -std=c++14 -O3
# TEST_FLAGS  := -std=c++14 -O0 -Wall -coverage -fprofile-arcs -ftest-coverage # DEBUGGING

# DEBUGGING
# CFLAGS      := -std=c++14 -g -O0 
#-Wall -Wextra -Wno-unused-variable -Wno-unused-parameter -fno-omit-frame-pointer -fsanitize=address
# RELEASE
CFLAGS := -std=c++14 -O3 -march=native #-Wall -Wextra

# LIB         := -framework Accelerate
TEST_LIB    := -framework Accelerate

INC         := -isystem include
TEST_INC    := -isystem . -isystem include -I/usr/local/include -isystem src -isystem test

GTEST_DIR   := include/googletest
GTEST_BUILD := -isystem ${GTEST_DIR}/include -pthread
GTEST_LINK  := -isystem ${GTEST_DIR}/include include/libgtest.a
# GTEST_LINK  := -isystem ${GTEST_DIR}/include include/libgtest.a -fprofile-arcs -ftest-coverage # DEBUGGING

BUILD_GTEST := -isystem $(GTEST_DIR)/include -I$(GTEST_DIR) -pthread -c $(GTEST_DIR)/src/gtest-all.cc -o include/gtest-all.o

$(TARGET): $(OBJECTS)
	@echo " Linking..."
	@echo " $(CC) $^ -o $(TARGET) $(LIB)"; $(CC) $^ -o $(TARGET) $(LIB)

$(BUILDDIR)/%.o: $(SRCDIR)/%.$(SRCEXT)
	@echo " Building..."
	@mkdir -p $(@D)
	@echo " $(CC) $(CFLAGS) $(INC) -c -o $@ $<"; $(CC) $(CFLAGS) $(INC) -c -o $@ $<

test: $(TEST_TARGET)

$(TEST_TARGET): $(TEST_OBJECTS) $(OBJECTS_NO_MAIN)
	@echo " Linking..."
	@echo " $(CC) $^ -o $(TEST_TARGET) $(TEST_LIB) $(GTEST_LINK)"; $(CC) $^ -o $(TEST_TARGET) $(TEST_LIB) $(GTEST_LINK)

$(BUILDDIR)/%.o: $(TEST_DIR)/%.$(SRCEXT)
	@echo " Building..."
	@mkdir -p $(@D)
	@echo " $(CC) $(TEST_FLAGS) $(TEST_INC) $(GTEST_BUILD) -c -o $@ $<"; $(CC) $(TEST_FLAGS) $(TEST_INC) $(GTEST_BUILD) -c -o $@ $<

gtest:
	$(CC) $(BUILD_GTEST)
	ar -rv include/libgtest.a include/gtest-all.o

.PHONY: clean
clean:
	@echo " Cleaning..."; 
	@echo " $(RM) -r $(BUILDDIR) $(TARGET) $(TEST_TARGET)"; $(RM) -r $(BUILDDIR) $(TARGET) $(TEST_TARGET)
