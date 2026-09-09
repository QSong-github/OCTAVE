#!/bin/bash
# (1) 本会话作业一律 Nice=50000，让用户其它会话的新作业先调度；(2) 本会话运行核数 ≤ 109（115 留 6 核），只节流名为 xenmultiall 的阵列。
# 节流按实际核数：允许运行的大阵列任务总数 ALLOW = 在跑的大阵列任务数 + 空余核数/每任务核数；多个同名阵列时，每个阵列的 throttle = ALLOW − 其它阵列在跑的任务数，总和不会超过 ALLOW。
CAP=109; BIG=xenmultiall; PER=2
MINE="xenmultiall|xenanfix|xenpfcut|xenpfgpu|xenpfan|istarxen|istareval|istarhipt|fig1v3|fig1v3med|xenembfix|hestind|xenind"
while true; do
  for J in $(squeue -u $USER -h -t PD -o "%i %j" | grep -E " ($MINE)$" | awk '{print $1}' | sed 's/_\[.*//; s/_[0-9]*$//' | sort -u); do
    nice=$(scontrol show job $J 2>/dev/null | grep -o "Nice=[0-9-]*" | head -1 | cut -d= -f2); [ "${nice:-0}" -lt 50000 ] && scontrol update JobId=$J Nice=50000 2>/dev/null
  done
  R=$(squeue -u $USER -h -t R -o "%C" | paste -sd+ | bc 2>/dev/null); R=${R:-0}
  RB=$(squeue -u $USER -h -t R -n $BIG -o "%C" | wc -l); RBC=$(squeue -u $USER -h -t R -n $BIG -o "%C" | paste -sd+ | bc 2>/dev/null); RBC=${RBC:-0}
  FREE=$((CAP - R))
  if [ $FREE -ge 0 ]; then ALLOW=$((RB + FREE / PER)); else ALLOW=$((RB - ( -FREE + PER - 1) / PER)); fi
  [ $ALLOW -lt 0 ] && ALLOW=0
  PEND=$(squeue -u $USER -h -t PD -o "%j" | grep -v -c "^$BIG$"); [ $PEND -gt 0 ] && [ $ALLOW -gt 0 ] && ALLOW=$((ALLOW - 1))
  LINE=""
  for J in $(squeue -u $USER -h -n $BIG -o "%i" | grep -o "^[0-9]*_\[" | tr -d "_[" | sort -u); do
    RBJ=$(squeue -u $USER -h -t R -n $BIG -o "%i" | grep -c "^${J}_"); T=$((ALLOW - (RB - RBJ))); [ $T -lt 0 ] && T=0
    scontrol update JobId=$J ArrayTaskThrottle=$T 2>/dev/null; LINE="$LINE $J:$T"
  done
  echo "$(date +%H:%M) running=$R cores (big=$RB tasks/$RBC cores, others=$((R - RBC))) → allow=$ALLOW throttles:$LINE"
  n=$(squeue -u $USER -h -n $BIG | wc -l); [ "$n" -eq 0 ] && { echo "大阵列结束，调速器退出"; break; }
  sleep 60
done
