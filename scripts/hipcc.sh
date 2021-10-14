#!/bin/bash

/opt/rocm/bin/hipcc -D__HIPCC__ -D__hcc_major__=3 -D__hcc_minor__=10 "$@"
