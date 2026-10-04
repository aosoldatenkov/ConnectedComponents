#!/usr/bin/env bash

c++ -O3 -Wall -shared -std=c++17 -fPIC \
    $(python3 -m pybind11 --includes) \
    groups.cpp \
    -o groups$(python3-config --extension-suffix)
