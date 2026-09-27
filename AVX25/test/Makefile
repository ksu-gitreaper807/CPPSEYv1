# Synthetic test harness for the PSE detector.
#
# Builds the UNMODIFIED detector source (../june/loadv102_DD.cpp — the latest
# iteration) against API-compatible fake FFmpeg/X11 headers, replacing the
# screen-capture device with a procedural frame generator (scenarios.cpp).
#
#   make            — build build/detector_harness
#   make clean
#
# See README.md for usage and the evaluation matrix.

CXX      ?= g++
CXXFLAGS ?= -O3 -mavx2 -mfma -mbmi2 -mpopcnt -std=c++17 -pthread -Wall -I fakeff
SRC      = ../june/loadv102_DD.cpp
OBJ      = fakeff_impl.cpp scenarios.cpp
BIN      = build/detector_harness

$(BIN): $(SRC) $(OBJ) $(wildcard fakeff/*.h)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -o $@ $(SRC) $(OBJ)

clean:
	rm -rf build results

.PHONY: clean
