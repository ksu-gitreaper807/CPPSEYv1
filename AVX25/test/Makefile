# Synthetic test harness for the PSE detector.
#
# Builds the UNMODIFIED detector source (../june/loadv102_DD.cpp — the latest
# iteration) against API-compatible fake FFmpeg/X11 headers, replacing the
# screen-capture device with a procedural frame generator (scenarios.cpp).
#
#   make            — build build/detector_harness
#   make clean
#
# See ../README.md for usage and benchmark methodology.

CXX      ?= g++
CXXFLAGS ?= -O3 -mavx2 -mfma -mbmi2 -mpopcnt -std=c++20 -pthread -Wall -I fakeff -DPSE_TEST_HARNESS=1
SRC      = ../june/loadv102_DD.cpp
OBJ      = fakeff_impl.cpp scenarios.cpp
BIN      = build/detector_harness

all: $(BIN) build/sensitivity_tests

$(BIN): Makefile $(SRC) $(OBJ) $(wildcard ../june/*.hpp) $(shell find fakeff -name '*.h')
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -o $@ $(SRC) $(OBJ)

build/sensitivity_tests: sensitivity_tests.cpp $(wildcard ../june/*.hpp)
	@mkdir -p build
	$(CXX) -O2 -std=c++20 -Wall -I ../june -o $@ sensitivity_tests.cpp

clean:
	rm -rf build results

.PHONY: all clean
