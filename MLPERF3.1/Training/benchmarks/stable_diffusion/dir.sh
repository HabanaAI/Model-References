#!/bin/bash

path="./results"

str="-"

declare -A dir_cr_times

for dir in "$path"/*/;
do
  if [ -d "$dir" ] && [[ "$dir" == *"$str"* ]] ; then
    cr_time=$(stat -c %Y "$dir")
    dir_cr_times["$dir"]=$cr_time
  fi
done

sorted_dirs=($(for dir in "${!dir_cr_times[@]}"; do
  echo "$dir_cr_times[$dir]"
done | sort -n -r -k2 | awk '{print $1}'))

for dir in "${sorted_dirs[@]}"; do
  d_p=${dir%"]"}
  d_p=${d_p#"["}
  echo ${d_p}
done
