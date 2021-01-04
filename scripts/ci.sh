#!/bin/sh
# Copyright (c) 2020 Agenium Scale
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

###############################################################################
# Argument parsing

if [ "$3" == "" ]; then
  echo "ERROR: usage: $0 JOBS_FILE REMOTE_DIR NSIMD_NSTOOLS_CHECKOUT_LATER"
  exit 1
fi

JOBS_FILE="`realpath $1`"
REMOTE_DIR="$2"
W_REMOTE_DIR="`echo ${REMOTE_DIR} | tr / \\\\`"
NSIMD_NSTOOLS_CHECKOUT_LATER="$3"

cd `dirname $0`
#set -x
set -e

###############################################################################
# Init

SSH="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
         -o LogLevel=error"
SCP="scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
         -o LogLevel=error"
GIT_URL=`git remote get-url origin`
GIT_BRANCH=`git rev-parse --abbrev-ref HEAD`
TMP_DIR="${PWD}/../_ci"
ONE_LINER_C="${PWD}/../scripts/one-liner.c"
SSHJOB_C="${PWD}/../nstools/sshjob/sshjob.c"

# Empty tmp directory
if [ -f "${JOBS_FILE}" ]; then
  rm -rf "${TMP_DIR}"
  mkdir -p "${TMP_DIR}"
fi

###############################################################################
# Build jobs scripts

function write_end_of_script() {
  if [ "${3}" == "Windows" ]; then
    cat >>"${1}" <<-EOF
	  echo Finished

	) |& one-liner ci-nsimd-${2}-output.txt ci-nsimd-${2}-one-liner.txt

	echo Finished >ci-nsimd-${2}-one-liner.txt
	EOF
  else
    cat >>"${1}" <<-EOF
	  echo Finished

	) |& ./one-liner ci-nsimd-${2}-output.txt ci-nsimd-${2}-one-liner.txt

	echo Finished >ci-nsimd-${2}-one-liner.txt
	EOF
  fi
}

if [ -f "${JOBS_FILE}" ]; then

  CURRENT_JOB=""
  DESC=""
  REMOTE_HOST="Linux"
  
  while read -r line; do
  
    # Empty lines
    if [ "`echo ${line} | sed 's/[ \t]*//g'`" == "" ]; then
      continue
    fi

    # Comments
    if [ "`echo ${line} | cut -c 1`" == "#" ]; then
      continue
    fi
  
    if [ "`echo ${line} | cut -c 1`" == "-" ]; then
      if [ "${REMOTE_HOST}" == "Windows" ]; then
        echo "  `echo ${line} | cut -c 2-`" >>"${CURRENT_JOB}"
      else
        echo "  `echo ${line} | cut -c 2-` && \\" >>"${CURRENT_JOB}"
      fi
      echo >>"${CURRENT_JOB}"
    else
      if [ "${CURRENT_JOB}" != "" ]; then
        write_end_of_script "${CURRENT_JOB}" "${DESC}" "${REMOTE_HOST}"
      fi
      ADDR=`echo ${line} | sed -e 's/(.*)//g' -e 's/  *//g'`
      DESC=`echo ${line} | sed -e 's/.*(//g' -e 's/).*//g'`
      REMOTE_HOST=`echo ${ADDR} | head -c 3`
      if [ "${REMOTE_HOST}" == "WIN" ]; then
        CURRENT_JOB="${TMP_DIR}/${ADDR}--${DESC}.bat" # <-- this must be before
        ADDR="`echo ${ADDR} | tail -c +4`"            # <-- this
        REMOTE_HOST="Windows"
        cat >"${CURRENT_JOB}" <<-EOF
	@echo off

	setlocal
	pushd "%~dp0"
	
        set NSIMD_NSTOOLS_CHECKOUT_LATER="${NSIMD_NSTOOLS_CHECKOUT_LATER}"

	if exist ci-nsimd-${DESC} rd /Q /S ci-nsimd-${DESC}
	if exist ci-nsimd-${DESC}-output.txt del /F /Q ci-nsimd-${DESC}-output.txt
	if exist ci-nsimd-${DESC}-one-liner.txt del /F /Q ci-nsimd-${DESC}-one-liner.txt

	git clone ${GIT_URL} ci-nsimd-${DESC}
	git -C ci-nsimd-${DESC} checkout ${GIT_BRANCH}

	(
	  pushd ci-nsimd-${DESC}

	EOF
      else
        CURRENT_JOB="${TMP_DIR}/${ADDR}--${DESC}.sh"
        REMOTE_HOST="Linux"
        cat >"${CURRENT_JOB}" <<-EOF
	#!/bin/sh
	
	cd \`dirname \$0\`
	set -e

        export NSIMD_NSTOOLS_CHECKOUT_LATER="${NSIMD_NSTOOLS_CHECKOUT_LATER}"

	rm -rf ci-nsimd-${DESC}
	rm -f ci-nsimd-${DESC}-output.txt
	rm -f ci-nsimd-${DESC}-one-liner.txt

	git clone ${GIT_URL} ci-nsimd-${DESC}
	git -C ci-nsimd-${DESC} checkout ${GIT_BRANCH}

	(
	  cd ci-nsimd-${DESC} && \\

	EOF
      fi
    fi

  done <"${JOBS_FILE}"

  if [ "${CURRENT_JOB}" != "" ]; then
    write_end_of_script "${CURRENT_JOB}" "${DESC}" "${REMOTE_HOST}"
  fi

fi

###############################################################################
# Launch jobs

if [ -f "${JOBS_FILE}" ]; then

  echo "-- NSIMD CI"
  echo "-- "
  echo "-- Initialization:"
  
  for job in "${TMP_DIR}"/*.sh "${TMP_DIR}"/*.bat; do
    ADDR=`basename ${job} .sh | sed 's/--.*//g'`
    DESC=`basename ${job} .sh | sed 's/.*--//g'`
    REMOTE_HOST=`echo ${ADDR} | head -c 3`
    if [ "${REMOTE_HOST}" == "WIN" ]; then
      REMOTE_HOST="Windows"
      ADDR="`echo ${ADDR} | tail -c +4`"
    else
      REMOTE_HOST="Linux"
    fi
    echo "-- Found new job: ${DESC}"
    echo "--   Remote machine will be: ${ADDR}"
    if [ "${REMOTE_HOST}" == "Windows" ]; then
      echo "--   Working directory will be: ${W_REMOTE_DIR}"
      ${SSH} ${ADDR} if not exist ${W_REMOTE_DIR} md ${W_REMOTE_DIR}
    else
      echo "--   Working directory will be: ${REMOTE_DIR}"
      ${SSH} ${ADDR} mkdir -p ${REMOTE_DIR}
    fi
    echo "--   Launching commands"
    if [ "${REMOTE_HOST}" == "Windows" ]; then
      ${SCP} ${job} ${ADDR}:${W_REMOTE_DIR}
      ${SCP} ${ONE_LINER_C} ${ADDR}:${W_REMOTE_DIR}
      ${SCP} ${SSHJOB_C} ${ADDR}:${W_REMOTE_DIR}
      ${SSH} ${ADDR} "cd ${W_REMOTE_DIR} & \
                      cl /Ox /W3 /D_CRT_SECURE_NO_WARNINGS one-liner.c"
      ${SSH} ${ADDR} "cd ${W_REMOTE_DIR} & \
                      cl /Ox /W3 /D_CRT_SECURE_NO_WARNINGS sshjob.c"
      ${SSH} ${ADDR} ${W_REMOTE_DIR}\\sshjob run ${W_REMOTE_DIR}\\${job} \
             >${TMP_DIR}/ci-nsimd-${DESC}-pid.txt
    else
      ${SCP} ${job} ${ADDR}:${REMOTE_DIR}
      ${SCP} ${ONE_LINER_C} ${ADDR}:${REMOTE_DIR}
      ${SCP} ${SSHJOB_C} ${ADDR}:${REMOTE_DIR}
      ${SSH} ${ADDR} "cd ${REMOTE_DIR} && cc -O2 one-liner.c -o one-liner"
      ${SSH} ${ADDR} "cd ${REMOTE_DIR} && cc -O2 sshjob.c -o sshjob"
      ${SSH} ${ADDR} ${REMOTE_DIR}/sshjob run ${REMOTE_DIR}/${job} \
             >${TMP_DIR}/ci-nsimd-${DESC}-pid.txt
    fi
  done

  sleep 2

fi

###############################################################################
# Build associative arrays

REMOTE_HOST_A=""
ADDR_A=""
DESC_A=""
ONE_LINER_A=""
KILL_COMMAND_A=""
LOG_A=""
N=0

for job in "${TMP_DIR}/"*.sh; do
  ADDR=`basename ${job} .sh | sed 's/--.*//g'`
  REMOTE_HOST=`echo ${ADDR} | head -c 3`
  if [ "${REMOTE_HOST}" == "WIN" ]; then
    REMOTE_HOST="Windows"
    ADDR="`echo ${ADDR} | tail -c +4`"
    LOG="${W_REMOTE_DIR}\\ci-nsimd-${DESC}-output.txt"
    ONE_LINER="${W_REMOTE_DIR}\\ci-nsimd-${DESC}-one-liner.txt"
    KILL_COMMAND="${W_REMOTE_DIR}\\sshjob kill \
                  `cat ${TMP_DIR}/ci-nsimd-${DESC}-pid.txt"
  else
    REMOTE_HOST="Linux"
    ONE_LINER="${REMOTE_DIR}/ci-nsimd-${DESC}-one-liner.txt"
    LOG="${REMOTE_DIR}/ci-nsimd-${DESC}-output.txt"
    KILL_COMMAND="${REMOTE_DIR}/sshjob kill \
                  `cat ${TMP_DIR}/ci-nsimd-${DESC}-pid.txt"
  fi
  DESC=`basename ${job} .sh | sed 's/.*--//g'`
  PID="${TMP_DIR}/ci-nsimd-${DESC}-pid.txt"

  ADDR_A="${ADDR_A}${ADDR}:"
  DESC_A="${DESC_A}${DESC}:"
  ONE_LINER_A="${ONE_LINER_A}${ONE_LINER}:"
  KILL_COMMAND_A="${KILL_COMMAND_A}${KILL_COMMAND}:"
  LOG_A="${LOG_A}${LOG}:"
  REMOTE_HOST_A="${REMOTE_HOST_A}${REMOTE_HOST}:"

  N=`expr ${N} + 1`
done

get_a() {
  echo ${1} | cut -f${2} -d':'
}
 
###############################################################################
# Monitor jobs (main event loop)

if [ -d "${JOBS_FILE}" ]; then
  TMP_DIR="${JOBS_FILE}"
fi

trap "stty echo icanon; exit 0" SIGINT
stty -echo -icanon
clear
key=""
selected=1

echo2() {
 printf "%-${COLUMNS}s" " "
 echo -e "\r${1}"
}

while true; do
  if [ "${selected}" -gt "${N}" ]; then
    selected=${N}
  fi
  if [ "${selected}" -lt "1" ]; then
    selected=1
  fi

  # Display part
  tput cup 0 0
  key=""
  echo2
  echo2 "[q] quit         [D] download outputs and quit  [T] kill all jobs"
  echo2 "[j] select next  [k] select previous            [t] kill selected job"
  echo2 "                 [d] see selected job log"
  echo2
  for i in `seq 1 ${N}`; do
    ADDR=`get_a ${ADDR_A} ${i}`
    DESC=`get_a ${DESC_A} ${i}`
    ONE_LINER=`get_a ${ONE_LINER_A} ${i}`
    REMOTE_HOST=`get_a ${REMOTE_HOST_A} ${i}`
    if [ "${REMOTE_HOST}" == "Windows" ]; then
      STATUS=`${SSH} ${ADDR} "if exist ${ONE_LINER} type ${ONE_LINER}" \
                     </dev/null || true`
    else
      STATUS=`${SSH} ${ADDR} "[ -f ${ONE_LINER} ] && cat ${ONE_LINER}" \
                     </dev/null || true`
    fi
    if [ "${i}" == "${selected}" ]; then
      echo2 "++++  ${i}: ${ADDR}, ${DESC}  ++++"
    else
      echo2 "${i}: ${ADDR}, ${DESC}"
    fi
    W=`expr ${COLUMNS} - 4`
    echo2 "    `echo ${STATUS} | cut -c 1-${W}`"
    echo2
    read -t 0.01 -n 1 key || true
    if [ "${key}" != "" ]; then
      break
    fi
  done

  # Keyboard input part
  if [ "${key}" == "" ]; then
    read -t 0.01 -n 1 key || true
  fi
  if [ "${key}" == "" ]; then
    continue
  fi
  if [ "${key}" == "q" ]; then
    break
  fi
  if [ "${key}" == "j" ]; then
    selected=`expr ${selected} + 1`
    continue
  fi
  if [ "${key}" == "k" ]; then
    selected=`expr ${selected} - 1`
    continue
  fi
  if [ "`echo ${key} | grep [123456789]`" != "" ]; then
    selected=${key}
    continue
  fi
  if [ "${key}" == "t" ]; then
    ADDR=`get_a ${ADDR_A} ${selected}`
    KILL_COMMAND=`get_a ${KILL_SCRIPT_A} ${selected}`
    ${SSH} ${ADDR} ${KILL_COMMAND}
    clear
    continue
  fi
  if [ "${key}" == "T" ]; then
    clear
    echo
    echo "Terminating every job..."
    echo
    for i in `seq 1 ${N}`; do
      ADDR=`get_a ${ADDR_A} ${i}`
      KILL_COMMAND=`get_a ${KILL_SCRIPT_A} ${i}`
      ${SSH} ${ADDR} ${KILL_COMMAND}
    done
    echo
    echo "...done"
    echo
    break
  fi
  if [ "${key}" == "d" ]; then
    ADDR=`get_a ${ADDR_A} ${selected}`
    LOG=`get_a ${LOG_A} ${selected}`
    ${SCP} ${ADDR}:${LOG} ${TMP_DIR}/log.txt
    less ${TMP_DIR}/log.txt
    clear
    continue
  fi
  if [ "${key}" == "D" ]; then
    clear
    echo
    echo "Downloading every log..."
    echo
    for i in `seq 1 ${N}`; do
      ADDR=`get_a ${ADDR_A} ${i}`
      LOG=`get_a ${LOG_A} ${i}`
      ${SCP} ${ADDR}:${LOG} ${TMP_DIR}
    done
    echo
    echo "...done"
    echo
    break
  fi
done

stty echo icanon
exit 0
