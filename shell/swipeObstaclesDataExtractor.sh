#!/bin/bash

if [ $# -ne 1 ]; then
    echo "args number must be 1!"
    exit 1
fi

rostopic echo -b $1 -p /ndt_pose/pose/position > pose.csv
echo "ndt_pose is extracted"
rostopic echo -b $1 -p /ypspur_ros/odom/pose/pose/position > odom.csv
echo "odom is extracted"
rostopic echo -b $1 -p /ypspur_ros/odom/twist/twist/linear/x > vel.csv
echo "velocity is extracted"
rostopic echo -b $1 -p /shifted_info > shift.csv
echo "shift info is extracted"
rostopic echo -b $1 -p /joy > joy.csv
echo "joy is extracted"
rostopic echo -b $1 -p /detected_obstacles > obstacle.csv
echo "obstacle is extracted"
