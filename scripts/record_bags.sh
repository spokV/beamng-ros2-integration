#!/bin/bash

# Source ROS2 environment
source /opt/ros/jazzy/setup.bash
source /home/spok/ros2_ws/install/setup.bash

ros2 bag record -o \
/media/spok/data/beamng/rosbags/offroad_drive_$(date +%Y%m%d_%H%M%S) \
/vehicles/ego/sensors/front_cam/colour \
/vehicles/ego/sensors/state \
/vehicles/ego/sensors/electrics \
/vehicles/ego/sensors/imu \
/vehicles/ego/sensors/lidar \
/vehicles/ego/sensors/powertrain \
/vehicles/ego/sensors/damage \
/standalone_vision_llm/driving_prompt \
--max-bag-size 0