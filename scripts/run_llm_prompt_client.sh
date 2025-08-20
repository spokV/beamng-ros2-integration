#!/bin/bash

source /opt/ros/jazzy/setup.bash
source /home/spok/ros2_ws/install/setup.bash

python3 -m beamng_ros2.examples.vision_llm_client \
    --config /home/spok/ros2_ws/src/beamng-ros2-integration/examples/vision_prompt_config.json \
    --interval 10.0