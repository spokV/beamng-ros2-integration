"""
Standalone Vision-Language-Action (VLA) prompt generator.

This script independently:
1. Subscribes to camera topic for visual input
2. Subscribes to vehicle state topic for vehicle context  
3. Generates LLM prompts every X seconds combining vision + state
4. Publishes contextual driving prompts

Usage:
    # Make sure to source your ROS2 environment first:
    # conda activate jazzy_env
    # cd /home/spok/ros2_ws
    # source install/setup.bash
    # python3 src/beamng-ros2-integration/standalone_vision_llm.py
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
import argparse
from sensor_msgs.msg import Image
from beamng_msgs.msg import StateSensor, DrivingPrompt
from geometry_msgs.msg import Point
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
import time
import json
import threading
from queue import Queue, Empty
import base64
import io
from PIL import Image as PILImage
import numpy as np
from typing import Optional, Dict, Any


class StandaloneVisionLLM(Node):
    def __init__(self, prompt_interval=10.0, config_file=None):
        super().__init__('vision_llm')
        
        # Configuration
        self.prompt_interval = prompt_interval  # Generate prompt every N seconds
        self.config = self._load_config(config_file)
        
        # LLM configuration (default or from config file)
        # Check both possible config structures
        prompt_config = self.config.get("driving_prompt", self.config.get("prompt_publisher_config", {}))
        self.llm_config = prompt_config.get("llm_config", {
            "provider": "openrouter",
            "api_key": "",
            "model": "anthropic/claude-3-haiku",
            "temperature": 0.7,
            "max_tokens": 300
        })
        
        # Debug logging for API key
        api_key = self.llm_config.get("api_key", "")
        if api_key:
            self.get_logger().debug(f"Loaded API key from config: {api_key[:10]}...")
        else:
            self.get_logger().warn("No API key found in config file - LLM calls will fail")
        
        # State tracking
        self.latest_camera_image = None
        self.latest_vehicle_state = None
        self.camera_callback_count = 0
        self.state_callback_count = 0
        self.last_prompt_time = 0.0
        self.first_image_received = False
        self.last_image_timestamp = None
        self.first_llm_response_received = False
        
        # LLM processing
        self.llm_request_queue = Queue()
        self.llm_response_queue = Queue()
        self.current_llm_prompt = None
        self._llm_worker_running = True
        self.llm_thread = threading.Thread(target=self._llm_worker, daemon=True)
        self.llm_thread.start()
        
        self._setup_subscribers()
        self._setup_publisher()
        # Don't setup timer yet - wait for first image
        self.prompt_timer = None
        
        self.get_logger().debug(f"Standalone Vision-LLM system started! Waiting for first image...")
        self.get_logger().debug(f"Will generate prompts every {self.prompt_interval}s after first image received")
        self.get_logger().debug(f"Subscribed to camera: /vehicles/ego/sensors/front_cam/colour")
        self.get_logger().debug(f"Subscribed to state: /vehicles/ego/sensors/state")
        self.get_logger().debug(f"Will publish to: /vision_llm/driving_prompt")
        if config_file:
            self.get_logger().debug(f"Loaded config from: {config_file}")
        
        # Remove periodic status check for cleaner output
        # self.status_timer = self.create_timer(5.0, self._status_check)
    
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        if not config_file:
            return {}
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            self.get_logger().debug(f"Successfully loaded config from {config_file}")
            self.get_logger().debug(f"Config keys: {list(config.keys())}")
            return config
        except FileNotFoundError:
            self.get_logger().error(f"Config file not found: {config_file}")
            return {}
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Invalid JSON in config file {config_file}: {e}")
            return {}
        except Exception as e:
            self.get_logger().error(f"Error loading config file {config_file}: {e}")
            return {}
    
    def _setup_subscribers(self):
        """Setup ROS2 subscribers for camera and vehicle state."""
        # QoS profile for reliable communication
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Camera subscriber (using the working topic from our tests)
        self.camera_subscriber = self.create_subscription(
            Image,
            "/vehicles/ego/sensors/front_cam/colour",
            self._camera_callback,
            qos_profile
        )
        self.get_logger().debug("Created camera subscriber for /vehicles/ego/sensors/front_cam/colour")
        
        # Vehicle state subscriber
        self.state_subscriber = self.create_subscription(
            StateSensor,
            "/vehicles/ego/sensors/state",
            self._state_callback,
            qos_profile
        )
        self.get_logger().debug("Created state subscriber for /vehicles/ego/sensors/state")
    
    def _setup_publisher(self):
        """Setup publisher for driving prompts."""
        self.prompt_publisher = self.create_publisher(
            DrivingPrompt,
            "/vision_llm/driving_prompt",
            1
        )
        self.get_logger().debug("Created prompt publisher at /vision_llm/driving_prompt")
    
    def _setup_timer(self):
        """Setup timer for periodic prompt generation."""
        self.prompt_timer = self.create_timer(self.prompt_interval, self._generate_prompt)
        self.get_logger().debug(f"Created prompt generation timer (every {self.prompt_interval}s)")
    
    def _camera_callback(self, msg):
        """Callback for camera image messages."""
        self.latest_camera_image = msg
        self.camera_callback_count += 1
        self.last_image_timestamp = msg.header.stamp
        
        # Start timer after receiving first image and ensuring we have vehicle state
        if not self.first_image_received and self.latest_vehicle_state is not None:
            self.first_image_received = True
            self.get_logger().debug("First image received with vehicle state available! Requesting first LLM prompt...")
            # Submit first LLM request but don't start timer yet
            context = self._prepare_llm_context()
            self.llm_request_queue.put(context)
        
        # Check for first LLM response (to break deadlock)
        if self.first_image_received and not self.first_llm_response_received:
            self._check_for_first_llm_response()
        
        if self.camera_callback_count <= 3 or self.camera_callback_count % 20 == 0:
            self.get_logger().debug(
                f"[CAMERA] Received image #{self.camera_callback_count}: "
                f"{msg.width}x{msg.height}, encoding={msg.encoding}"
            )
    
    def _state_callback(self, msg):
        """Callback for vehicle state messages."""
        self.latest_vehicle_state = msg
        self.state_callback_count += 1
        
        # Check if we can start timer now (have both image and state)
        if not self.first_image_received and self.latest_camera_image is not None:
            self.first_image_received = True
            self.get_logger().debug("Vehicle state received with camera image available! Requesting first LLM prompt...")
            # Submit first LLM request but don't start timer yet
            context = self._prepare_llm_context()
            self.llm_request_queue.put(context)
        
        if self.state_callback_count <= 3 or self.state_callback_count % 20 == 0:
            speed = np.sqrt(msg.velocity.x**2 + msg.velocity.y**2 + msg.velocity.z**2)
            self.get_logger().debug(
                f"[STATE] Received state #{self.state_callback_count}: "
                f"pos=({msg.position.x:.1f}, {msg.position.y:.1f}, {msg.position.z:.1f}), "
                f"speed={speed:.2f} m/s"
            )
    
    def _generate_prompt(self):
        """Generate and publish a new driving prompt."""
        try:
            # Check if we have the required data
            if self.latest_camera_image is None:
                self.get_logger().debug("No camera image available - skipping prompt generation")
                return
            
            if self.latest_vehicle_state is None:
                self.get_logger().debug("No vehicle state available - skipping prompt generation")
                return
            
            # Prepare context for LLM
            context = self._prepare_llm_context()
            
            # Try to get recent LLM response
            try:
                llm_response = self.llm_response_queue.get_nowait()
                if llm_response:
                    self.current_llm_prompt = llm_response
                    
                    # If this is the first LLM response, start the timer
                    if not self.first_llm_response_received:
                        self.first_llm_response_received = True
                        self._setup_timer()
                        self.get_logger().debug("First LLM response received! Starting prompt generation timer.")
                    
                    # Log the complete LLM response
                    self.get_logger().debug("="*80)
                    self.get_logger().debug("[LLM] COMPLETE PROMPT RECEIVED:")
                    self.get_logger().debug("="*80)
                    self.get_logger().debug(f"Task Description: {llm_response.get('task_description', 'N/A')}")
                    self.get_logger().debug(f"Current Objective: {llm_response.get('current_objective', 'N/A')}")
                    self.get_logger().debug(f"Driving Instructions:")
                    for i, instruction in enumerate(llm_response.get('driving_instructions', []), 1):
                        self.get_logger().debug(f"  {i}. {instruction}")
                    self.get_logger().debug(f"Required Skills: {llm_response.get('required_skills', [])}")
                    self.get_logger().debug("="*80)
            except Empty:
                pass  # No new response available
            
            # Always submit new LLM request for next interval (respecting the timer)
            self.llm_request_queue.put(context)
            
            # Always publish if we have an LLM response (respecting the timer interval)
            if self.current_llm_prompt:
                self._publish_prompt()
            elif not self.first_llm_response_received:
                self.get_logger().debug("Waiting for first LLM response before publishing any prompts...")
            else:
                self.get_logger().debug("Timer interval reached but no LLM response available yet")
            
        except Exception as e:
            self.get_logger().error(f"Error generating prompt: {e}")
    
    def _status_check(self):
        """Periodic status check to help debug issues."""
        status_msg = f"[STATUS] Camera: {self.camera_callback_count} msgs, State: {self.state_callback_count} msgs"
        
        if not self.first_image_received:
            status_msg += " | Waiting for first image+state"
        elif not self.first_llm_response_received:
            status_msg += " | Waiting for first LLM response"
        else:
            status_msg += f" | Active (Timer: {self.prompt_timer is not None})"
        
        if self.latest_camera_image:
            status_msg += f" | Last image: {self.latest_camera_image.width}x{self.latest_camera_image.height}"
        
        if self.latest_vehicle_state:
            speed = np.sqrt(self.latest_vehicle_state.velocity.x**2 + self.latest_vehicle_state.velocity.y**2 + self.latest_vehicle_state.velocity.z**2)
            status_msg += f" | Speed: {speed:.1f} m/s"
        
        self.get_logger().debug(status_msg)
    
    def _check_for_first_llm_response(self):
        """Check for first LLM response to break the deadlock."""
        try:
            llm_response = self.llm_response_queue.get_nowait()
            if llm_response:
                self.current_llm_prompt = llm_response
                self.first_llm_response_received = True
                self._setup_timer()
                self.get_logger().debug("First LLM response received! Starting prompt generation timer.")
                
                # Log the complete LLM response
                self.get_logger().debug("="*80)
                self.get_logger().debug("[LLM] COMPLETE PROMPT RECEIVED:")
                self.get_logger().debug("="*80)
                self.get_logger().debug(f"Task Description: {llm_response.get('task_description', 'N/A')}")
                self.get_logger().debug(f"Current Objective: {llm_response.get('current_objective', 'N/A')}")
                self.get_logger().debug(f"Driving Instructions:")
                for i, instruction in enumerate(llm_response.get('driving_instructions', []), 1):
                    self.get_logger().debug(f"  {i}. {instruction}")
                self.get_logger().debug(f"Required Skills: {llm_response.get('required_skills', [])}")
                self.get_logger().debug("="*80)
                
                # Publish the first prompt immediately
                self._publish_prompt()
        except Empty:
            pass  # No response available yet
    
    def _prepare_llm_context(self) -> Dict[str, Any]:
        """Prepare context for LLM including camera image and vehicle state."""
        # Vehicle state info
        state = self.latest_vehicle_state
        speed = np.sqrt(state.velocity.x**2 + state.velocity.y**2 + state.velocity.z**2)
        
        context = {
            "speed": speed,
            "position": [state.position.x, state.position.y, state.position.z],
            "orientation": {
                "dir": [state.dir.x, state.dir.y, state.dir.z],
                "up": [state.up.x, state.up.y, state.up.z],
                "front": [state.front.x, state.front.y, state.front.z]
            },
            "moving": speed > 1.0,
            "timestamp": time.time()
        }
        
        # Add camera image
        camera_image = self._encode_camera_image()
        if camera_image:
            context["camera_image"] = camera_image
            self.get_logger().debug(f"[VISION] Added camera image to LLM context ({len(camera_image)} chars)")
        
        return context
    
    def _encode_camera_image(self) -> Optional[str]:
        """Encode camera image to base64 for LLM."""
        try:
            if self.latest_camera_image is None:
                return None
            
            # Convert ROS Image to PIL Image (avoiding cv_bridge dependency issues)
            msg = self.latest_camera_image
            
            # ROS Image message data is already in the right format
            # msg.encoding = 'rgb8', msg.data contains raw pixel data
            if msg.encoding == 'rgb8':
                # Convert byte data to numpy array
                img_array = np.frombuffer(msg.data, dtype=np.uint8)
                img_array = img_array.reshape((msg.height, msg.width, 3))
                
                # Convert to PIL Image
                pil_img = PILImage.fromarray(img_array)
            else:
                self.get_logger().warn(f"Unsupported image encoding: {msg.encoding}")
                return None
            
            # Resize for faster processing  
            pil_img = pil_img.resize([256, 256], PILImage.LANCZOS)
            
            # Convert to base64
            buffer = io.BytesIO()
            pil_img.save(buffer, format='JPEG', quality=85)
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            self.get_logger().debug(f"[VISION] Encoded camera image: {msg.width}x{msg.height} -> 256x256, base64 ({len(img_base64)} chars)")
            return img_base64
            
        except Exception as e:
            self.get_logger().error(f"Error encoding camera image: {e}")
            return None
    
    def _llm_worker(self):
        """Background worker for LLM API calls."""
        self.get_logger().debug("[LLM_WORKER] Started LLM worker thread")
        while self._llm_worker_running:
            try:
                context = self.llm_request_queue.get(timeout=1.0)
                if context is None:  # Shutdown signal
                    break
                
                self.get_logger().debug("[LLM_WORKER] Processing LLM request...")
                response = self._call_llm(context)
                if response and self._llm_worker_running:
                    self.llm_response_queue.put(response)
                    self.get_logger().debug("[LLM_WORKER] LLM response queued successfully")
                else:
                    self.get_logger().warn("[LLM_WORKER] LLM call returned no response")
                
            except Empty:
                continue
            except Exception as e:
                if self._llm_worker_running:
                    self.get_logger().error(f"[LLM_WORKER] LLM worker error: {e}")
                    import traceback
                    self.get_logger().error(f"[LLM_WORKER] Traceback: {traceback.format_exc()}")
    
    def _call_llm(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Call LLM API with vision and state context."""
        try:
            self.get_logger().debug("[LLM_API] Starting LLM API call...")
            import openai
            
            api_key = self.llm_config.get("api_key", "")
            if not api_key:
                self.get_logger().error("[LLM_API] No API key available for LLM call!")
                return None
                
            self.get_logger().debug(f"[LLM_API] Making LLM call with API key: {api_key[:10]}...")
            self.get_logger().debug(f"[LLM_API] Model: {self.llm_config['model']}")
            
            client = openai.OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1"
            )
            
            system_prompt = """You are an expert off-road driving instructor providing contextual guidance for autonomous vehicle training in challenging terrain.

Generate concise, specific off-road driving prompts based on the current vehicle state and camera view. Focus on terrain assessment, traction management, and off-road navigation techniques.

Respond with a JSON object containing:
{
    "task_description": "Brief description of current off-road driving task",
    "current_objective": "Immediate off-road navigation objective", 
    "driving_instructions": ["instruction1", "instruction2", "instruction3"],
    "required_skills": ["skill1", "skill2"]
}

Focus on off-road specific elements:
- Terrain assessment from camera (rocks, slopes, loose surfaces, obstacles)
- Vehicle positioning and approach angles for obstacles
- Traction management (wheel placement, momentum control)
- Off-road techniques (hill climbing, descent control, rock crawling)
- Recovery strategies and safe navigation paths
- Surface conditions impact on vehicle dynamics"""
            
            user_prompt = f"""Current vehicle state:
- Speed: {context['speed']:.1f} m/s
- Position: {context['position']}
- Orientation vectors:
  - Direction: {context['orientation']['dir']}
  - Up: {context['orientation']['up']} 
  - Front: {context['orientation']['front']}
- Moving: {context['moving']}

Camera view: Provided as image input

Generate appropriate off-road driving guidance for this situation. Consider both the vehicle orientation vectors and what you can see in the front camera view to provide specific terrain-appropriate instructions."""
            
            messages = [
                {"role": "system", "content": system_prompt}
            ]
            
            if "camera_image" in context and context["camera_image"]:
                # Vision-enabled message
                user_message = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{context['camera_image']}"
                            }
                        }
                    ]
                }
            else:
                # Text-only message
                user_message = {"role": "user", "content": user_prompt}
            
            messages.append(user_message)
            
            self.get_logger().debug("[LLM_API] Sending request to OpenRouter API...")
            response = client.chat.completions.create(
                model=self.llm_config["model"],
                messages=messages,
                temperature=self.llm_config["temperature"],
                max_tokens=self.llm_config["max_tokens"],
                extra_headers={
                    "HTTP-Referer": "beamng-ros2-integration",
                    "X-Title": "BeamNG VLA Training"
                }
            )
            
            self.get_logger().debug("[LLM_API] Received response from API, parsing...")
            parsed_response = self._parse_llm_response(response.choices[0].message.content)
            self.get_logger().debug("[LLM_API] Response parsed successfully")
            return parsed_response
            
        except Exception as e:
            self.get_logger().error(f"[LLM_API] LLM API error: {e}")
            import traceback
            self.get_logger().error(f"[LLM_API] Traceback: {traceback.format_exc()}")
            return None
    
    def _parse_llm_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Parse LLM response into structured format."""
        try:
            # Try to extract JSON from response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                # Clean control characters that cause JSON parsing issues
                import re
                json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)
                return json.loads(json_str)
            else:
                # Fallback: create structured response from text
                lines = response_text.strip().split('\n')
                return {
                    "task_description": lines[0] if lines else "Navigate safely",
                    "current_objective": lines[1] if len(lines) > 1 else "Drive carefully",
                    "driving_instructions": lines[2:5] if len(lines) > 2 else ["Observe surroundings", "Maintain control"],
                    "required_skills": ["general_driving"]
                }
                
        except Exception as e:
            self.get_logger().error(f"Failed to parse LLM response: {e}")
            return None
    
    def _publish_prompt(self):
        """Publish the current driving prompt."""
        try:
            # Only use LLM-generated prompts, no fallbacks
            if not self.current_llm_prompt:
                self.get_logger().error("_publish_prompt called without LLM response!")
                return
            
            prompt_data = self.current_llm_prompt
            self.get_logger().debug("[PROMPT] Publishing LLM-generated prompt")
            
            # Calculate current vehicle speed
            state = self.latest_vehicle_state
            current_speed = np.sqrt(state.velocity.x**2 + state.velocity.y**2 + state.velocity.z**2)
            
            # Create ROS message
            msg = DrivingPrompt()
            # Use image timestamp if available, otherwise current time
            if self.last_image_timestamp:
                msg.header.stamp = self.last_image_timestamp
            else:
                msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'ego'
            
            # Use config values if available, otherwise use defaults or LLM response
            # Check both possible config structures
            driving_prompt_config = self.config.get("driving_prompt", self.config.get("prompt_publisher_config", {}))
            
            msg.task_description = prompt_data.get('task_description', 
                                   driving_prompt_config.get('task_description', 'Navigate safely'))
            msg.scenario_name = driving_prompt_config.get('scenario_name', 'johnson_valley_vla_training')
            msg.level_name = driving_prompt_config.get('level_name', 'johnson_valley')
            msg.weather_conditions = driving_prompt_config.get('weather_conditions', 'clear')
            msg.time_of_day = driving_prompt_config.get('time_of_day', 'day')
            msg.current_objective = prompt_data.get('current_objective', 
                                   driving_prompt_config.get('current_objective', 'Drive safely'))
            msg.traffic_density = driving_prompt_config.get('traffic_density', 'low')
            msg.road_type = driving_prompt_config.get('road_type', 'off_road')
            msg.visibility_conditions = driving_prompt_config.get('visibility_conditions', 'good')
            msg.driving_instructions = prompt_data.get('driving_instructions', 
                                      driving_prompt_config.get('driving_instructions', ['Drive carefully']))
            
            # Handle target_location from config
            target_loc = driving_prompt_config.get('target_location', [0.0, 0.0, 0.0])
            msg.target_location = Point(x=float(target_loc[0]), y=float(target_loc[1]), z=float(target_loc[2]))
            
            msg.difficulty_level = driving_prompt_config.get('difficulty_level', 'medium')
            msg.required_skills = prompt_data.get('required_skills', 
                                  driving_prompt_config.get('required_skills', ['general_driving']))
            
            # Add vehicle speed (assuming there's a field for it in DrivingPrompt)
            # Note: If the message doesn't have a speed field, we'll add it to the task description
            try:
                msg.vehicle_speed = float(current_speed)
            except AttributeError:
                # If no speed field exists, add it to the task description
                msg.task_description = f"{msg.task_description} (Speed: {current_speed:.1f} m/s)"
            
            # Publish
            self.prompt_publisher.publish(msg)
            self.get_logger().debug(f"[PROMPT] Published: {msg.task_description[:60]}... (objective: {msg.current_objective})")
            
        except Exception as e:
            self.get_logger().error(f"Error publishing prompt: {e}")
    
    def destroy_node(self):
        """Cleanup when shutting down."""
        self._llm_worker_running = False
        self.llm_request_queue.put(None)  # Shutdown signal
        if self.llm_thread.is_alive():
            self.llm_thread.join(timeout=2.0)
        super().destroy_node()


def main(args=None):
    parser = argparse.ArgumentParser(description='Standalone Vision-LLM for BeamNG')
    parser.add_argument('--interval', '-i', type=float, default=10.0,
                       help='Prompt generation interval in seconds (default: 10.0)')
    parser.add_argument('--model', '-m', type=str, default='anthropic/claude-3-haiku',
                       help='LLM model to use (default: anthropic/claude-3-haiku)')
    parser.add_argument('--config', '-c', type=str,
                       help='Path to JSON configuration file')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging (shows camera/state details)')
    parser.add_argument('--log-level', type=str, default='info',
                       choices=['debug', 'info', 'warn', 'error'],
                       help='Set log level (default: info)')
    
    parsed_args = parser.parse_args(args)
    
    rclpy.init()
    
    node = StandaloneVisionLLM(prompt_interval=parsed_args.interval, config_file=parsed_args.config)
    
    # Update model if specified (command line overrides config file)
    if parsed_args.model != 'anthropic/claude-3-haiku':
        node.llm_config["model"] = parsed_args.model
    
    # Set log level based on CLI argument (debug flag overrides)
    if parsed_args.debug:
        log_level = rclpy.logging.LoggingSeverity.DEBUG
    else:
        log_level_map = {
            'debug': rclpy.logging.LoggingSeverity.DEBUG,
            'info': rclpy.logging.LoggingSeverity.INFO,
            'warn': rclpy.logging.LoggingSeverity.WARN,
            'error': rclpy.logging.LoggingSeverity.ERROR
        }
        log_level = log_level_map[parsed_args.log_level]
    
    node.get_logger().set_level(log_level)
    
    node.get_logger().debug(f"Starting with prompt interval: {parsed_args.interval}s, model: {parsed_args.model}")
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().debug("Shutting down standalone vision-LLM system...")
    except ExternalShutdownException:
        node.get_logger().debug("External shutdown requested")
    except rclpy._rclpy_pybind11.RCLError:
        node.get_logger().debug("RCL context shutdown - terminating gracefully")
    finally:
        try:
            node.destroy_node()
        except:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except:
            pass


if __name__ == '__main__':
    main()