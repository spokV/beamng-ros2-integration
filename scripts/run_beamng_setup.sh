#!/bin/bash

# BeamNG ROS2 Automation Script
# Runs BeamNG binary, builds bridge, and starts scenario in sequence

set -e  # Exit on any error

# Configuration
BEAMNG_HOST="127.0.0.1"
BEAMNG_PORT="25252"  # Default BeamNG -tcom port
BEAMNG_BINARY="/home/spok/Downloads/BeamNG.tech.v0.36.4.0/BinLinux/BeamNG.tech.x64"
SCENARIO_CONFIG="/config/scenarios/johnson_valley.json"
VISION_CONFIG="/home/spok/ros2_ws/src/beamng-ros2-integration/configs/vision_prompt_config.json"
PUBLISH_RATE_SEC="0.01"  # Publish rate in seconds (0.01 = 100Hz, 0.033 = 30Hz, 0.1 = 10Hz)
MAX_WAIT_TIME=60
CHECK_INTERVAL=2

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if BeamNG is running
check_beamng_running() {
    pgrep -f "BeamNG" > /dev/null 2>&1
}

# Function to detect which port BeamNG is actually using
detect_beamng_port() {
    local beamng_pid=$(pgrep -f "BeamNG" | head -1)
    if [ ! -z "$beamng_pid" ]; then
        # Check common BeamNG ports
        for port in 25252 64256 64257; do
            if netstat -tlnp 2>/dev/null | grep -q ":$port.*$beamng_pid/"; then
                echo $port
                return 0
            fi
        done
    fi
    return 1
}

# Function to check if BeamNG is ready for connections
check_beamng_ready() {
    # Simple TCP connection test is sufficient
    timeout 2 bash -c "echo >/dev/tcp/$BEAMNG_HOST/$BEAMNG_PORT" >/dev/null 2>&1
    return $?
}

# Function to wait for BeamNG to be ready
wait_for_beamng() {
    print_status "Waiting for BeamNG to be ready..."
    local elapsed=0
    local last_status=""
    
    # Try to auto-detect the actual port BeamNG is using
    if check_beamng_running; then
        local detected_port=$(detect_beamng_port)
        if [ ! -z "$detected_port" ] && [ "$detected_port" != "$BEAMNG_PORT" ]; then
            print_warning "BeamNG detected on port $detected_port (expected $BEAMNG_PORT)"
            print_status "Updating BEAMNG_PORT to $detected_port"
            BEAMNG_PORT="$detected_port"
        fi
    fi
    
    while [ $elapsed -lt $MAX_WAIT_TIME ]; do
        # Check if BeamNG process is still running
        if ! check_beamng_running; then
            print_error "BeamNG process has stopped running"
            return 1
        fi
        
        # Get current status for better feedback
        local current_status=""
        local cpu_usage
        cpu_usage=$(ps -o %cpu -p $(pgrep -f "BeamNG" | head -1) 2>/dev/null | tail -1 | tr -d ' ')
        
        # Check if any BeamNG port is listening
        local port_open=false
        for port in 25252 64256 64257; do
            if timeout 1 bash -c "echo >/dev/tcp/$BEAMNG_HOST/$port" >/dev/null 2>&1; then
                if [ "$port" != "$BEAMNG_PORT" ]; then
                    print_warning "BeamNG detected on port $port (updating from $BEAMNG_PORT)"
                    BEAMNG_PORT="$port"
                fi
                port_open=true
                break
            fi
        done
        
        if $port_open; then
            if [ ! -z "$cpu_usage" ] && [ "${cpu_usage%.*}" -gt 50 ]; then
                current_status="port open, still loading (CPU: ${cpu_usage}%)"
            else
                current_status="ready on port $BEAMNG_PORT"
            fi
        else
            if [ ! -z "$cpu_usage" ] && [ "${cpu_usage%.*}" -gt 30 ]; then
                current_status="initializing (CPU: ${cpu_usage}%) - waiting for TCP port"
            else
                current_status="process running but no TCP port yet"
            fi
        fi
        
        # Print status if changed
        if [ "$current_status" != "$last_status" ]; then
            echo ""
            print_status "BeamNG status: $current_status"
            last_status="$current_status"
        fi
        
        # Check if fully ready
        if check_beamng_ready; then
            echo ""
            print_success "BeamNG is ready for connections!"
            return 0
        fi
        
        sleep $CHECK_INTERVAL
        elapsed=$((elapsed + CHECK_INTERVAL))
        echo -n "."
    done
    
    echo ""
    print_error "Timeout waiting for BeamNG to be ready after ${MAX_WAIT_TIME}s"
    print_status "Current BeamNG log (last 10 lines):"
    tail -n 10 /tmp/beamng.log 2>/dev/null || echo "No log available"
    return 1
}

# Function to stop existing ROS2 bridge processes
stop_existing_bridges() {
    print_status "Stopping existing BeamNG bridge processes..."
    
    # Kill any running beamng_bridge processes
    pkill -f "beamng_bridge" || true
    
    # Wait a moment for processes to terminate
    sleep 2
    
    # Force kill if still running
    pkill -9 -f "beamng_bridge" || true
    
    print_success "Stopped existing bridge processes"
}

# Function to activate conda environment and source ROS2
source_ros_environment() {
    print_status "Activating conda jazzy_env environment..."
    
    # Initialize conda for bash using the conda.sh file
    source /home/spok/miniconda/etc/profile.d/conda.sh
    
    # Activate the jazzy_env conda environment
    if conda activate jazzy_env; then
        print_success "Activated conda jazzy_env environment"
    else
        print_error "Failed to activate conda jazzy_env environment"
        exit 1
    fi
    
    print_status "Sourcing ROS2 environment..."
    
    if [ -f "/opt/ros/jazzy/setup.bash" ]; then
        source /opt/ros/jazzy/setup.bash
    elif [ -f "/opt/ros/humble/setup.bash" ]; then
        source /opt/ros/humble/setup.bash
    elif [ -f "/opt/ros/foxy/setup.bash" ]; then
        source /opt/ros/foxy/setup.bash
    else
        print_error "No ROS2 installation found"
        exit 1
    fi
    
    # Source workspace
    if [ -f "/home/spok/ros2_ws/install/setup.bash" ]; then
        source /home/spok/ros2_ws/install/setup.bash
    else
        print_warning "Workspace not built yet, will build now"
    fi
    
    # Ensure conda site-packages is in PYTHONPATH
    export PYTHONPATH="/home/spok/miniconda/envs/jazzy_env/lib/python3.12/site-packages:$PYTHONPATH"
    
    print_success "ROS2 environment sourced"
}

# Function to build the workspace
build_workspace() {
    print_status "Building BeamNG ROS2 workspace..."
    
    cd /home/spok/ros2_ws
    
    # Ensure we're still in the conda environment
    
    # Build the workspace
    if colcon build --packages-select beamng_msgs beamng_ros2 beamng_bringup beamng_agent beamng_teleop_keyboard >/dev/null 2>&1; then
        print_success "Workspace built successfully"
        
        # Source the newly built workspace
        source /home/spok/ros2_ws/install/setup.bash
        print_status "Workspace sourced"
    else
        print_error "Failed to build workspace"
        print_error "Stopping all processes and exiting..."
        cleanup
        exit 1
    fi
}

# Function to start BeamNG if not running
start_beamng() {
    if check_beamng_running; then
        print_success "BeamNG is already running"
        wait_for_beamng
        return 0
    fi
    
    print_status "Starting BeamNG from: $BEAMNG_BINARY"
    
    # Check if binary exists
    if [ ! -f "$BEAMNG_BINARY" ]; then
        print_error "BeamNG binary not found at: $BEAMNG_BINARY"
        print_error "Please update BEAMNG_BINARY path in the script"
        exit 1
    fi
    
    # Make sure binary is executable
    chmod +x "$BEAMNG_BINARY"
    
    # Start BeamNG in background with required arguments
    print_status "Launching BeamNG with -tcom and -tcom-listen-ip $BEAMNG_HOST"
    print_status "Note: BeamNG may take time to start or crash on first attempt"
    
    cd "$(dirname "$BEAMNG_BINARY")"
    
    # Try starting BeamNG with TCP communication enabled
    print_status "Starting with TCP communication server (-tcom)..."
    nohup "$BEAMNG_BINARY" -tcom -tcom-listen-ip "$BEAMNG_HOST" > /tmp/beamng.log 2>&1 &
    
    BEAMNG_PID=$!
    echo $BEAMNG_PID > /tmp/beamng.pid
    
    print_status "BeamNG launched with PID: $BEAMNG_PID"
    print_status "Log output: /tmp/beamng.log"
    
    # Give BeamNG some time to initialize
    sleep 10
    
    # Check if process is still running
    if ! kill -0 $BEAMNG_PID 2>/dev/null; then
        print_warning "BeamNG process crashed. Check log: /tmp/beamng.log"
        print_status "You may need to start BeamNG manually:"
        print_status "cd $(dirname "$BEAMNG_BINARY")"
        print_status "./BeamNG.tech.x64 -tcom -tcom-listen-ip $BEAMNG_HOST"
        print_status ""
        print_status "Waiting for BeamNG to be started manually..."
        
        # Wait for manual start
        while ! check_beamng_running; do
            sleep 5
            echo -n "."
        done
        print_success "BeamNG detected as running"
    fi
    
    # Wait for BeamNG to be ready
    wait_for_beamng
}

# Function to start the bridge
start_bridge() {
    print_status "Starting BeamNG bridge..."
    
    # Ensure we're using the conda Python interpreter
    
    # Test beamngpy import before starting bridge
    if ! python -c "import beamngpy" 2>/dev/null; then
        print_error "beamngpy module not available in current environment"
        print_status "Try: conda activate jazzy_env && python -c 'import beamngpy'"
        exit 1
    fi
    
    # Start bridge in background with explicit Python path
    PYTHON_EXEC=$(which python)
    export PYTHONPATH="/home/spok/miniconda/envs/jazzy_env/lib/python3.12/site-packages:$PYTHONPATH"
    
    ros2 run beamng_ros2 beamng_bridge --ros-args -p host:=$BEAMNG_HOST -p port:=$BEAMNG_PORT -p launch:=false -p update_sec:=$PUBLISH_RATE_SEC 2>/dev/null &
    
    BRIDGE_PID=$!
    
    # Wait for bridge to start
    sleep 5
    
    # Check if bridge is still running
    if kill -0 $BRIDGE_PID 2>/dev/null; then
        print_success "Bridge started successfully (PID: $BRIDGE_PID)"
        echo $BRIDGE_PID > /tmp/beamng_bridge.pid
    else
        print_error "Bridge failed to start"
        print_status "Check if beamngpy is available: python -c 'import beamngpy'"
        exit 1
    fi
}

# Function to start the scenario
start_scenario() {
    print_status "Starting scenario: $SCENARIO_CONFIG"
    
    # Wait a moment for bridge to fully initialize
    sleep 3
    
    # Start the scenario
    if ros2 service call /beamng_bridge/start_scenario beamng_msgs/srv/StartScenario "{path_to_scenario_definition: '$SCENARIO_CONFIG'}" >/dev/null 2>&1; then
        print_success "Scenario started successfully"
    else
        print_error "Failed to start scenario"
        exit 1
    fi
}

# Function to start the LLM client
start_llm_client() {
    print_status "Starting LLM vision client..."
    
    # Wait for scenario to be fully loaded
    sleep 5
    
    # Start listening for the first prompt BEFORE starting the LLM client
    print_status "Setting up prompt listener before starting LLM client..."
    (
        if timeout 10 ros2 topic echo /vision_llm/driving_prompt --once >/dev/null 2>&1; then
            echo "FIRST_PROMPT_DETECTED" > /tmp/first_prompt_ready
        fi
    ) &
    PROMPT_LISTENER_PID=$!
    
    # Give the listener a moment to start
    sleep 2
    
    # Start the standalone vision LLM client with config file
    print_status "Starting LLM client with config: $VISION_CONFIG"
    python /home/spok/ros2_ws/src/beamng-ros2-integration/beamng_ros2/beamng_ros2/examples/vision_llm_client.py \
        --config "$VISION_CONFIG" --log-level warn &
    
    LLM_PID=$!
    
    # Wait for LLM client to start
    sleep 3
    
    # Check if LLM client is still running
    if kill -0 $LLM_PID 2>/dev/null; then
        print_success "LLM client started successfully (PID: $LLM_PID)"
        echo $LLM_PID > /tmp/llm_client.pid
        
        # Extract prompt interval from config file
        PROMPT_INTERVAL=$(grep '"prompt_interval_seconds"' "$VISION_CONFIG" | sed 's/.*"prompt_interval_seconds": *\([0-9.]*\).*/\1/' | head -1)
        if [ -z "$PROMPT_INTERVAL" ]; then
            PROMPT_INTERVAL="10"  # fallback default
        fi
        WAIT_TIME=$(echo "$PROMPT_INTERVAL * 2" | bc 2>/dev/null || echo "20")
        WAIT_TIME=${WAIT_TIME%.*}  # remove decimal part
        
        # Wait for first prompt detection
        print_status "Waiting for first LLM prompt (prompts every ${PROMPT_INTERVAL}s)..."
        for i in $(seq 1 $WAIT_TIME); do
            if [ -f /tmp/first_prompt_ready ]; then
                rm -f /tmp/first_prompt_ready
                print_success "First prompt detected!"
                break
            fi
            sleep 1
            echo -n "."
        done
        
        # Clean up listener if still running
        kill $PROMPT_LISTENER_PID 2>/dev/null || true
        
        echo ""
        print_success "System is ready for recording!"
        print_status ""
        print_status "Ready to record! You can now:"
        print_status "1. Run: ./scripts/record_bags.sh"  
        print_status "2. Start driving in BeamNG"
        print_status "3. Stop recording with Ctrl+C"
        print_status ""
    else
        print_error "LLM client failed to start - check for Python/config errors"
        # Clean up prompt listener
        kill $PROMPT_LISTENER_PID 2>/dev/null || true
        exit 1
    fi
}

# Function to wait for first LLM prompt publication
wait_for_first_prompt() {
    print_status "Monitoring /vision_llm/driving_prompt topic for first message..."
    
    # Simply wait for the first message with a generous timeout
    # This will block until a message is received or timeout occurs
    if timeout 30 ros2 topic echo /vision_llm/driving_prompt --once >/dev/null 2>&1; then
        print_success "First LLM prompt published successfully!"
        print_success "System is ready for recording!"
        print_status ""
        print_status "Ready to record! You can now:"
        print_status "1. Run: ./scripts/record_bags.sh"
        print_status "2. Start driving in BeamNG"
        print_status "3. Stop recording with Ctrl+C"
        print_status ""
        return 0
    else
        print_warning "Timeout waiting for first prompt (30s)"
        print_warning "LLM may be taking longer than expected - system should still work"
        print_status "You can proceed with recording, but check for LLM prompt errors"
        return 1
    fi
}

# Function to cleanup on exit
cleanup() {
    print_status "Cleaning up..."
    
    # Stop LLM client
    if [ -f /tmp/llm_client.pid ]; then
        local pid=$(cat /tmp/llm_client.pid)
        if kill -0 $pid 2>/dev/null; then
            print_status "Stopping LLM client (PID: $pid)"
            kill $pid
        fi
        rm -f /tmp/llm_client.pid
    fi
    
    # Stop bridge
    if [ -f /tmp/beamng_bridge.pid ]; then
        local pid=$(cat /tmp/beamng_bridge.pid)
        if kill -0 $pid 2>/dev/null; then
            print_status "Stopping bridge (PID: $pid)"
            kill $pid
        fi
        rm -f /tmp/beamng_bridge.pid
    fi
    
    # Stop BeamNG if we started it
    if [ -f /tmp/beamng.pid ]; then
        local pid=$(cat /tmp/beamng.pid)
        if kill -0 $pid 2>/dev/null; then
            print_status "Stopping BeamNG (PID: $pid)"
            kill $pid
            sleep 2
            # Force kill if still running
            kill -9 $pid 2>/dev/null || true
        fi
        rm -f /tmp/beamng.pid
    fi
}

# Set up cleanup trap
trap cleanup EXIT INT TERM

# Main execution
main() {
    print_status "Starting BeamNG ROS2 automation script..."
    print_status "Host: $BEAMNG_HOST, Port: $BEAMNG_PORT"
    print_status "Publish Rate: $PUBLISH_RATE_SEC sec ($(python3 -c "print(f'{1/$PUBLISH_RATE_SEC:.1f}')") Hz)"
    print_status "Scenario: $SCENARIO_CONFIG"
    print_status "Vision Config: $VISION_CONFIG"
    print_status ""
    
    # Step 1: Source ROS environment
    source_ros_environment
    
    # Step 2: Stop existing bridges
    stop_existing_bridges
    
    # Step 3: Build workspace
    build_workspace
    
    # Step 4: Start BeamNG (or wait for it)
    start_beamng
    
    # Step 5: Start the bridge
    start_bridge
    
    # Step 6: Start the scenario
    start_scenario
    
    # Step 7: Start the LLM client
    start_llm_client
    
    print_success "All systems started successfully!"
    print_status "Bridge is running in background (PID: $(cat /tmp/beamng_bridge.pid 2>/dev/null || echo 'unknown'))"
    print_status "LLM client is running in background (PID: $(cat /tmp/llm_client.pid 2>/dev/null || echo 'unknown'))"
    print_status "To stop the bridge, run: kill $(cat /tmp/beamng_bridge.pid 2>/dev/null || echo 'PID_NOT_FOUND')"
    print_status "Or press Ctrl+C to stop this script and cleanup"
    
    # Keep script running to maintain bridge and LLM client
    print_status "Press Ctrl+C to stop..."
    while true; do
        sleep 10
        # Check if bridge is still running
        if [ -f /tmp/beamng_bridge.pid ]; then
            local pid=$(cat /tmp/beamng_bridge.pid)
            if ! kill -0 $pid 2>/dev/null; then
                print_error "Bridge process has stopped unexpectedly"
                exit 1
            fi
        fi
        # Check if LLM client is still running
        if [ -f /tmp/llm_client.pid ]; then
            local pid=$(cat /tmp/llm_client.pid)
            if ! kill -0 $pid 2>/dev/null; then
                print_warning "LLM client has stopped unexpectedly - restarting..."
                start_llm_client
            fi
        fi
    done
}

# Run main function
main "$@"