#!/bin/sh

# -----------------------------------------------------------------------------
# Init

INPUT="`realpath ${1}`"
TARGET="${2}"
cd `dirname $0`
ROOT="${PWD}/../build-local-ci"
mkdir -p "${ROOT}"
NPROC=`nproc`
if [ "${TARGET}" == "" ]; then
  TARGET="tests"
fi

# -----------------------------------------------------------------------------
# Make sure we have generated nsimd

python3 "${PWD}/../egg/hatch.py" -ltf

# -----------------------------------------------------------------------------
# Make sure we have the latest commit for nsconfig

NSCONFIG="${PWD}/../nstools/nsconfig/nsconfig"
NSTEST="${PWD}/../nstools/nsconfig/nstest"

[ -e "${NSCONFIG}" ] || ( export NSTOOLS_CHECKOUT_LAST_COMMIT=1 && \
                          bash "${PWD}/../scripts/setup.sh" )
[ -e "${NSTEST}" ] || ( export NSTOOLS_CHECKOUT_LAST_COMMIT=1 && \
                        bash "${PWD}/../scripts/setup.sh" )

# -----------------------------------------------------------------------------
# Parse input file

SIMD_EXTS=""

while read -r line; do

  # Empty lines
  if [ "`echo ${line} | sed 's/[ \t]*//g'`" == "" ]; then
    continue
  fi

  # Comments
  if [ "`echo ${line} | cut -c 1`" == "#" ]; then
    continue
  fi

  # New architectures
  if [ "`echo ${line} | cut -c 1`" == "[" ]; then
    SIMD_EXTS="`echo ${line} | sed -e 's/[][,]/ /g'`"
    for s in ${SIMD_EXTS}; do
      echo '#!/bin/bash' >"${ROOT}/run-${s}.sh"
      echo >>"${ROOT}/run-${s}.sh"
      echo 'cd `dirname $0`' >>"${ROOT}/run-${s}.sh"
      echo "mkdir -p ${s}" >>"${ROOT}/run-${s}.sh"
      echo "cd ${s}" >>"${ROOT}/run-${s}.sh"
      echo >>"${ROOT}/run-${s}.sh"
    done
    continue
  fi

  # Standard line (part of a script)
  if [ "${SIMD_EXTS}" != "" ]; then
    for s in ${SIMD_EXTS}; do
      echo ${line} | sed -e "s,SIMD_EXT,${s},g" \
                         -e "s,SRC_DIR,${PWD}/..,g" \
                         -e "s,NSCONFIG,${NSCONFIG},g" \
                         -e "s,NSTEST,${NSTEST},g" \
                         -e "s,NPROC,${NPROC},g" \
                         -e "s,TARGET,${TARGET},g" \
                         >>"${ROOT}/run-${s}.sh"
    done
  fi
 
done <"${INPUT}"

# -----------------------------------------------------------------------------
# Compile all tests

for i in ${ROOT}/*.sh; do
  ( bash ${i} || true ) | tee ${i}.log
done
