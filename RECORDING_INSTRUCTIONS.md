# BeamNG ROS2 Recording Instructions

Follow these steps in order to record bag files with vision LLM prompts:

## Prerequisites
- BeamNG.tech running
- ROS2 environment set up
- All dependencies installed

## Recording Steps

### 1. Start BeamNG Bridge
```bash
ros2 run beamng_ros2 beamng_bridge
```

### 2. Start Scenario
```bash
ros2 service call /beamng_bridge/start_scenario beamng_msgs/srv/StartScenario "{path_to_scenario_definition: '/config/scenarios/johnson_valley.json'}"
```

### 3. Start Vision LLM Client
```bash
./scripts/run_llm_prompt_client.sh
```

### 4. Start Recording
```bash
./scripts/record_bags.sh
```

## Recorded Topics
- `/vehicles/ego/sensors/front_cam/colour` - Front camera images
- `/vehicles/ego/sensors/state` - Vehicle state information
- `/vehicles/ego/sensors/electrics` - Electrical system data
- `/vehicles/ego/sensors/imu` - IMU data
- `/vehicles/ego/sensors/lidar` - LiDAR data
- `/vehicles/ego/sensors/powertrain` - Powertrain data
- `/vehicles/ego/sensors/damage` - Damage information
- `/standalone_vision_llm/driving_prompt` - Vision LLM generated prompts

## Output Location
Bag files are saved to: `/media/spok/data/beamng/rosbags/offroad_drive_YYYYMMDD_HHMMSS/`