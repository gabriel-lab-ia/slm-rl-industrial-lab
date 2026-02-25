#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: $0 {fold|yaw|extend|zero|custom "[a,b,c,d,e,f]"}"
  exit 1
fi

MODE="$1"
case "$MODE" in
  fold)
    POS='[0.0, 0.8, -0.8, 0.0, 0.0, 0.0]'
    ;;
  yaw)
    POS='[0.9, 0.0, 0.0, 0.0, 0.0, 0.0]'
    ;;
  extend)
    POS='[0.0, -0.5, 0.7, 0.0, 0.0, 0.0]'
    ;;
  zero)
    POS='[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]'
    ;;
  custom)
    if [ $# -lt 2 ]; then
      echo "usage: $0 custom '[a,b,c,d,e,f]'"
      exit 1
    fi
    POS="$2"
    ;;
  *)
    echo "invalid mode: $MODE"
    exit 1
    ;;
esac

set +u
source /opt/ros/jazzy/setup.bash
set -u
ros2 topic pub --once /arm_1/joint_command sensor_msgs/msg/JointState "{position: $POS}"
