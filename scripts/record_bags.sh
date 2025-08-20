#!/bin/bash

# Instructions for recording bags:
# 1. Run BeamNG bridge: ros2 run beamng_ros2 beamng_bridge
# 2. Start scenario: ros2 service call /beamng_bridge/start_scenario beamng_msgs/srv/StartScenario "{path_to_scenario_definition: '/config/scenarios/johnson_valley.json'}"
# 3. Run vision LLM client: ./scripts/run_llm_prompt_client.sh
# 4. Run this script to start recording

# Configuration
HOME_DIR="/home/spok"

# Source ROS2 environment
source /opt/ros/jazzy/setup.bash
source "$HOME_DIR/ros2_ws/install/setup.bash"

# Parse configuration paths from run_beamng_setup.sh
SCRIPT_DIR="$(dirname "$0")"
SCENARIO_CONFIG=$(grep '^SCENARIO_CONFIG=' "$SCRIPT_DIR/run_beamng_setup.sh" | cut -d'"' -f2)
VISION_CONFIG=$(grep '^VISION_CONFIG=' "$SCRIPT_DIR/run_beamng_setup.sh" | cut -d'"' -f2)

# Extract scenario name from config path for folder organization
SCENARIO_NAME=$(basename "$SCENARIO_CONFIG" .json)

# Handle scenario config path - it starts with /config but needs full path
if [[ "$SCENARIO_CONFIG" = "/config"* ]]; then
    # Path starts with /config - prepend workspace root
    SCENARIO_FULL_PATH="$HOME_DIR/ros2_ws/src/beamng-ros2-integration/beamng_ros2$SCENARIO_CONFIG"
elif [[ "$SCENARIO_CONFIG" = /* ]]; then
    # Full absolute path
    SCENARIO_FULL_PATH="$SCENARIO_CONFIG"
else
    # Relative path - prepend workspace root
    SCENARIO_FULL_PATH="$HOME_DIR/ros2_ws/src/beamng-ros2-integration/beamng_ros2/$SCENARIO_CONFIG"
fi

# Extract POI name from scenario config for subfolder
POI_NAME=$(grep '"_poi"' "$SCENARIO_FULL_PATH" | sed 's/.*"_poi": *"\([^"]*\)".*/\1/' | head -1)
if [ -z "$POI_NAME" ]; then
    POI_NAME="default_location"
fi

# Create output directory structure: base/scenario_name/poi_name/bag_folder
OUTPUT_BASE="/media/spok/data/beamng/rosbags"
OUTPUT_DIR="$OUTPUT_BASE/$SCENARIO_NAME/$POI_NAME"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BAG_NAME="offroad_drive_$TIMESTAMP"
mkdir -p "$OUTPUT_DIR"

# Function to copy configs after recording
copy_configs() {
    BAG_DIR="$OUTPUT_DIR/$BAG_NAME"
    if [ -d "$BAG_DIR" ]; then
        echo "Copying configuration files to recording directory..."
        cp "$SCENARIO_FULL_PATH" "$BAG_DIR/scenario_config.json" 2>/dev/null || echo "Failed to copy scenario config"
        cp "$VISION_CONFIG" "$BAG_DIR/vision_config.json" 2>/dev/null || echo "Failed to copy vision config"
        echo "Configuration files copied to: $BAG_DIR"
    fi
}

# Set trap to copy configs when script exits
trap copy_configs EXIT

# Start recording first (this will create the bag directory)
echo "Starting bag recording: $BAG_NAME"
echo "Config files will be copied when recording stops."
ros2 bag record -o \
"$OUTPUT_DIR/$BAG_NAME" \
/tf \
/vehicles/ego/sensors/front_cam/colour \
/vehicles/ego/sensors/state \
/vehicles/ego/sensors/electrics \
/vehicles/ego/sensors/imu \
/vehicles/ego/sensors/lidar \
/vehicles/ego/sensors/powertrain \
/vehicles/ego/sensors/damage \
/vision_llm/driving_prompt \
--max-bag-size 0