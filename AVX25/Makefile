CXX ?= g++
CPPFLAGS += $(shell pkg-config --cflags libavdevice libavformat libavcodec libswscale libavutil x11 xrandr)
CXXFLAGS ?= -O3 -g
CXXFLAGS += -std=c++20 -mavx2 -mfma -mbmi2 -mpopcnt -pthread -Wall -Wextra
LDLIBS += $(shell pkg-config --libs libavdevice libavformat libavcodec libswscale libavutil x11 xrandr)

.PHONY: all check benchmark
all: detector

detector: june/loadv102_DD.cpp $(wildcard june/*.hpp) pse_portal_capture.py
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) june/loadv102_DD.cpp -o $@ $(LDLIBS)

check:
	$(MAKE) -C test
	./test/build/sensitivity_tests
	python3 -m unittest discover -s test -p 'test_*.py' -v
	$(MAKE) detector
	PSE_NO_PROTECTION=1 PSE_TEST_CAPTURE=1 PSE_MAX_FRAMES=20 ./detector

benchmark:
	$(MAKE) -C test
	python3 test/benchmark.py
