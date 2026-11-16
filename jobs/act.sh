# Sourced by job scripts : conda activate with retries and a python 3 check.
# On 2026-11-11 the conda CLI failed to import packaging on two busy nodes. vhdprep then ran the system python 2 and xi57 task 0 died.
source /path/to/miniconda3/etc/profile.d/conda.sh
# Some nodes failed to import packaging; exclude such nodes in your scheduler if this happens.
act() {
  local i
  for i in 1 2 3 4 5; do
    conda activate "$1" && [ "${CONDA_DEFAULT_ENV:-}" = "$1" ] && python -c "import sys; assert sys.version_info >= (3, 8)" && return 0
    echo "conda activate $1 failed (try $i)" >&2; sleep 30
  done
  echo "conda activate $1 failed 5 times" >&2; exit 3
}
