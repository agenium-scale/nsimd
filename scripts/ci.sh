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

if [ "$1" == "" ]; then
  echo "ERROR: usage: $0 JOBS_FILE REMOTE_DIR"
  exit 1
fi

if [ "$2" == "" ]; then
  echo "ERROR: usage: $0 JOBS_FILE REMOTE_DIR"
  exit 1
fi

JOBS_FILE="`realpath $1`"
REMOTE_DIR="$2"

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
    PID=`${SSH} ${ADDR} "bash -c 'echo \$PPID' ; bash ${rjob} </dev/null \
                         1>/dev/null 2>/dev/null &"`
    echo "--   PID = ${PID}"
    echo ${PID} >"${TMP_DIR}/`basename ${job} .sh`.pid"
  done

fi

sleep 2

###############################################################################
# Monitor jobs

if [ -d "${JOBS_FILE}" ]; then
  TMP_DIR="${JOBS_FILE}"
fi

clear
key=""

while true; do
  tput cup 0 0
  echo
  echo "[q] quit    [d] download outputs and quit    [k] kill all jobs"
  echo
  for job in "${TMP_DIR}/"*.sh; do
    ADDR=`basename ${job} .sh | sed 's/--.*//g'`
    DESC=`basename ${job} .sh | sed 's/.*--//g'`
    FILENAME="${REMOTE_DIR}/ci-nsimd-${DESC}-one-liner.txt"
    ONE_LINER=`${SSH} ${ADDR} "[ -f ${FILENAME} ] && cat ${FILENAME}" \
                      </dev/null || echo`
    printf "%-${COLUMNS}s" " "
    echo -e "\r${ADDR}, ${DESC}"
    printf "%-${COLUMNS}s" " "
    W=`expr ${COLUMNS} - 4`
    echo -e "\r    `echo ${ONE_LINER} | cut -c 1-${W}`"
    echo
  done
  read -t 1 -n 1 key || true
  if [ "${key}" == "" ]; then
    continue
  fi
  if [ "${key}" == "q" ]; then
    exit 0
  fi
  if [ "${key}" == "k" ]; then
    for pid in "${TMP_DIR}"/*.pid; do
      ADDR=`basename ${job} .sh | sed 's/--.*//g'`
      DESC=`basename ${job} .sh | sed 's/.*--//g'`
      PID=`cat ${pid}`
      ${SSH} ${ADDR} "pkill -P ${PID}"
    done
    exit 0
  fi
  if [ "${key}" == "d" ]; then
    for job in "${TMP_DIR}/"*.sh; do
      ADDR=`basename ${job} .sh | sed 's/--.*//g'`
      DESC=`basename ${job} .sh | sed 's/.*--//g'`
      FILENAME="${REMOTE_DIR}/ci-nsimd-${DESC}-output.txt"
      ${SCP} ${ADDR}:${FILENAME} ${TMP_DIR}
    done
    exit 0
  fi
  key=""
done
