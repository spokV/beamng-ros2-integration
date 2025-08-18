#!/bin/bash

# Instructions for recording bags:
# 1. Run BeamNG bridge: ros2 run beamng_ros2 beamng_bridge
# 2. Start scenario: ros2 service call /beamng_bridge/start_scenario beamng_msgs/srv/StartScenario "{path_to_scenario_definition: '/config/scenarios/johnson_valley.json'}"
# 3. Run vision LLM client: ./scripts/run_llm_prompt_client.sh
# 4. Run this script to start recording

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