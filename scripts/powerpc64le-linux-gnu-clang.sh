#!/bin/bash

clang --target=powerpc64le-linux-gnu \
      -I/usr/powerpc64le-linux-gnu/include/c++/8/powerpc64le-linux-gnu "$@"
