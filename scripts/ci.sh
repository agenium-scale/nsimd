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

# Empty tmp directory
if [ -f "${JOBS_FILE}" ]; then
  rm -rf "${TMP_DIR}"
  mkdir -p "${TMP_DIR}"
fi

###############################################################################
# Build jobs scripts

function write_end_of_script() {
  cat >>"${1}" <<-EOF
	  echo "Finished"

	) |& while read -r line; do
	  echo "\$line" >>ci-nsimd-${2}-output.txt
	  NEW=\`date +%s\`
	  if [ "\${OLD}" != "\${NEW}" ]; then
	    echo "\$line" >ci-nsimd-${2}-one-liner.txt
	    OLD=\${NEW}
	  fi
	done 

	echo "Finished" >ci-nsimd-${2}-one-liner.txt
	EOF
}

if [ -f "${JOBS_FILE}" ]; then

  CURRENT_JOB=""
  DESC=""
  
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
      echo "  `echo ${line} | cut -c 2-` && \\" >>"${CURRENT_JOB}"
      echo >>"${CURRENT_JOB}"
    else
      if [ "${CURRENT_JOB}" != "" ]; then
        write_end_of_script "${CURRENT_JOB}" "${DESC}"
      fi
      ADDR=`echo ${line} | sed -e 's/(.*)//g' -e 's/  *//g'`
      DESC=`echo ${line} | sed -e 's/.*(//g' -e 's/).*//g'`
      CURRENT_JOB="${TMP_DIR}/${ADDR}--${DESC}.sh"
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

	OLD=\`date +%s\`
	
	(
	  cd ci-nsimd-${DESC} && \\

	EOF
    fi

  done <"${JOBS_FILE}"

  if [ "${CURRENT_JOB}" != "" ]; then
    write_end_of_script "${CURRENT_JOB}" "${DESC}"
  fi

fi

###############################################################################
# Launch jobs

if [ -f "${JOBS_FILE}" ]; then

  echo "-- NSIMD CI"
  echo "-- "
  echo "-- Initialization:"
  
  for job in "${TMP_DIR}"/*.sh; do
    ADDR=`basename ${job} .sh | sed 's/--.*//g'`
    DESC=`basename ${job} .sh | sed 's/.*--//g'`
    echo "-- Found new job: ${DESC}"
    echo "--   Remote machine will be: ${ADDR}"
    echo "--   Working directory will be: ${REMOTE_DIR}"
    ${SSH} ${ADDR} mkdir -p ${REMOTE_DIR}
    echo "--   Launching commands"
    ${SCP} ${job} ${ADDR}:${REMOTE_DIR}
    rjob="${REMOTE_DIR}/`basename ${job}`"
    PID=`${SSH} ${ADDR} "echo \\\$PPID ; bash ${rjob} </dev/null \
                         1>/dev/null 2>/dev/null &"`
    echo "--   PID = ${PID}"
    KILL_SCRIPT="`basename ${job} .sh`.ks"
    cat >>"${TMP_DIR}/${KILL_SCRIPT}" <<-EOF
	#!/bin/sh

	list_children() {
	  local children=\`ps -o pid= --ppid "\${1}"\`
	  for pid in \${children}; do
	    list_children "\${pid}"
	  done
	  echo " \${children} "
        }

        kill -TERM \`list_children ${PID}\` ${PID}
	EOF
    ${SCP} ${TMP_DIR}/${KILL_SCRIPT} ${ADDR}:${REMOTE_DIR}
  done

  sleep 2

fi

###############################################################################
# Build associative arrays

ADDR_A=""
DESC_A=""
ONE_LINER_A=""
KILL_SCRIPT_A=""
LOG_A=""
N=0

for job in "${TMP_DIR}/"*.sh; do
  ADDR=`basename ${job} .sh | sed 's/--.*//g'`
  DESC=`basename ${job} .sh | sed 's/.*--//g'`
  ONE_LINER="${REMOTE_DIR}/ci-nsimd-${DESC}-one-liner.txt"
  KILL_SCRIPT="`basename ${job} .sh`.ks"
  LOG="ci-nsimd-${DESC}-output.txt"

  ADDR_A="${ADDR_A}${ADDR}:"
  DESC_A="${DESC_A}${DESC}:"
  ONE_LINER_A="${ONE_LINER_A}${ONE_LINER}:"
  KILL_SCRIPT_A="${KILL_SCRIPT_A}${KILL_SCRIPT}:"
  LOG_A="${LOG_A}${LOG}:"

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
    STATUS=`${SSH} ${ADDR} "[ -f ${ONE_LINER} ] && cat ${ONE_LINER}" \
                   </dev/null || true`
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
    KILL_SCRIPT=`get_a ${KILL_SCRIPT_A} ${selected}`
    ${SSH} ${ADDR} "bash ${REMOTE_DIR}/${KILL_SCRIPT} </dev/null \
                    1>/dev/null 2>/dev/null &"
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
      KILL_SCRIPT=`get_a ${KILL_SCRIPT_A} ${i}`
      ${SSH} ${ADDR} "bash ${REMOTE_DIR}/${KILL_SCRIPT} </dev/null \
                      1>/dev/null 2>/dev/null &"
    done
    echo
    echo "...done"
    echo
    break
  fi
  if [ "${key}" == "d" ]; then
    ADDR=`get_a ${ADDR_A} ${selected}`
    LOG=`get_a ${LOG_A} ${selected}`
    ${SCP} ${ADDR}:${REMOTE_DIR}/${LOG} ${TMP_DIR}
    less ${TMP_DIR}/${LOG}
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
      ${SCP} ${ADDR}:${REMOTE_DIR}/${LOG} ${TMP_DIR}
    done
    echo
    echo "...done"
    echo
    break
  fi
done

stty echo icanon
exit 0
