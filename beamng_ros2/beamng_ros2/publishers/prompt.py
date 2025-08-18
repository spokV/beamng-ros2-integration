"""
Driving Prompt publisher for VLA training data annotation.
Publishes contextual information about the current driving scenario.
"""

from typing import Any, Dict, Type, Optional
from rclpy.time import Time
import beamng_msgs.msg as msgs
from geometry_msgs.msg import Point
from .base import VehiclePublisher
import json
import time
import threading
from queue import Queue, Empty
import base64
import io
from PIL import Image


class DrivingPromptPublisher(VehiclePublisher):
    """
    Driving prompt publisher for annotating trajectory data with contextual information
    for Vision-Language-Action (VLA) model training.
    
    This publisher generates structured prompts that describe:
    - The current driving scenario and objectives
    - Environmental conditions
    - Required driving behaviors
    - Contextual information for the VLA model
    
    Args:
        name: Name of the publisher
        config: Configuration dictionary containing prompt parameters
    """
    
    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        super().__init__()
        self.name = name
        self.config = config
        
        # Base configuration
        self.base_task_description = config.get("task_description", "Drive safely and follow traffic rules")
        self.scenario_name = config.get("scenario_name", "unknown")
        self.level_name = config.get("level_name", "unknown")
        self.weather_conditions = config.get("weather_conditions", "clear")
        self.time_of_day = config.get("time_of_day", "day")
        self.base_objective = config.get("current_objective", "Navigate to destination")
        self.traffic_density = config.get("traffic_density", "low")
        self.road_type = config.get("road_type", "highway")
        self.visibility_conditions = config.get("visibility_conditions", "good")
        self.base_instructions = config.get("driving_instructions", [])
        self.target_location = config.get("target_location", [0.0, 0.0, 0.0])
        self.difficulty_level = config.get("difficulty_level", "medium")
        self.required_skills = config.get("required_skills", ["lane_keeping", "speed_control"])
        
        # Dynamic prompt features
        self.enable_dynamic_prompts = config.get("enable_dynamic_prompts", True)
        self.last_speed = 0.0
        self.last_position = [0.0, 0.0, 0.0]
        
        # LLM-powered prompt generation
        self.use_llm_prompts = config.get("use_llm_prompts", False)
        self.llm_config = config.get("llm_config", {})
        self.llm_prompt_cache = {}
        self.llm_request_queue = Queue()
        self.llm_response_queue = Queue()
        self.llm_thread = None
        self.current_llm_prompt = None
        self._llm_worker_running = True
        self._publisher_stopped = False
        
        # Vision/camera integration
        self.use_camera_input = config.get("use_camera_input", False)
        self.camera_name = config.get("camera_name", "default")
        self.image_quality = config.get("image_quality", 85)
        self.resize_image = config.get("resize_image", [256, 256])  # Smaller for faster processing
        self.latest_camera_image = None  # Store latest image from ROS topic
        self.camera_subscriber = None  # ROS2 subscriber for camera topic
        
        # Prompt generation timing control
        self.prompt_interval = config.get("prompt_interval_seconds", 10.0)  # Generate new prompt every N seconds
        self.last_prompt_time = 0.0
        self.manual_trigger_enabled = config.get("enable_manual_trigger", True)
        self.manual_trigger_key = config.get("manual_trigger_key", "space")
        self.prompt_on_state_change = config.get("prompt_on_state_change", True)
        self.last_vehicle_state = None
        
        if self.use_llm_prompts:
            self._setup_llm_worker()
        
    def msg_type(self) -> Type:
        return msgs.DrivingPrompt
    
    def create_publisher(self, node):
        self._node = node
        self._publisher = node.create_publisher(
            self.msg_type(), f"~/driving_prompt", 1
        )
        
        # Create service for manual prompt generation
        if self.manual_trigger_enabled:
            from std_srvs.srv import Empty
            self._manual_trigger_service = node.create_service(
                Empty, 
                f"~/generate_prompt_now", 
                self._manual_trigger_callback
            )
            node.get_logger().info(f"Manual prompt trigger service available at ~/generate_prompt_now")
        
        # Create service to stop the prompt publisher
        from std_srvs.srv import Empty
        self._stop_service = node.create_service(
            Empty,
            f"~/stop_prompt_publisher",
            self._stop_callback
        )
        node.get_logger().info(f"Stop prompt publisher service available at ~/stop_prompt_publisher")
        
        # Set up camera subscriber if camera input is enabled
        if self.use_camera_input:
            node.get_logger().info(f"Camera input enabled - setting up subscriber (use_camera_input={self.use_camera_input})")
            self._setup_camera_subscriber(node)
        else:
            node.get_logger().info(f"Camera input disabled (use_camera_input={self.use_camera_input})")
    
    def _manual_trigger_callback(self, _, response):
        """Service callback for manual prompt generation."""
        print(f"[PROMPT] Manual trigger service called!")
        try:
            if self.use_llm_prompts and self.vehicle:
                # Force immediate prompt generation
                current_time = time.time()
                state_data = self.vehicle.sensors.poll()
                current_speed = (state_data['vel'][0]**2 + state_data['vel'][1]**2 + state_data['vel'][2]**2)**0.5
                current_pos = state_data['pos']
                
                vehicle_context = {
                    "speed": current_speed,
                    "position": current_pos,
                    "moving": current_speed > 1.0,
                    "distance_moved": 0.0  # Not relevant for manual trigger
                }
                
                # Add camera image if enabled
                if self.use_camera_input:
                    camera_image = self._encode_camera_image()
                    if camera_image:
                        vehicle_context["camera_image"] = camera_image
                        self._node.get_logger().info(f"[Manual Trigger] Added camera image to context")
                
                # Force LLM request
                self.llm_request_queue.put(vehicle_context)
                self.last_prompt_time = current_time
                self.last_vehicle_state = vehicle_context.copy()
                
                self._node.get_logger().info("[Manual Trigger] New LLM prompt requested")
            else:
                self._node.get_logger().warn("[Manual Trigger] LLM prompts not enabled or vehicle not available")
                
        except Exception as e:
            self._node.get_logger().error(f"[Manual Trigger] Error: {e}")
        
        return response
    
    def _stop_callback(self, _, response):
        """Service callback to stop the prompt publisher."""
        print(f"[PROMPT] Stop service called - shutting down prompt publisher")
        self._publisher_stopped = True
        self.remove()
        return response
    
    def _setup_camera_subscriber(self, node):
        """Set up ROS2 subscriber for camera images."""
        try:
            from sensor_msgs.msg import Image
            from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
            
            # Create QoS profile to match the publisher (RELIABLE, VOLATILE)
            qos_profile = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
                depth=1
            )
            
            # Subscribe to the camera topic - use relative path matching the camera publisher
            camera_topic = f"~/sensors/front_cam/colour"
            self.camera_subscriber = node.create_subscription(
                Image,
                camera_topic,
                self._camera_callback,
                qos_profile
            )
            node.get_logger().info(f"Camera subscriber created for topic: {camera_topic} with RELIABLE QoS")
            
            # Store node reference for debugging
            self._node_ref = node
            
            # Try to immediately check if we can see the topic
            import threading
            import time
            def delayed_check():
                time.sleep(2)  # Wait a bit for initialization
                try:
                    # Force a spin to process any pending callbacks
                    import rclpy
                    rclpy.spin_once(node, timeout_sec=0.1)
                    print(f"[CAMERA] Camera subscriber status check - callback count: {getattr(self, '_camera_callback_count', 0)}")
                except Exception as e:
                    print(f"[CAMERA] Error in delayed check: {e}")
            
            threading.Thread(target=delayed_check, daemon=True).start()
            
        except Exception as e:
            node.get_logger().error(f"Failed to create camera subscriber: {e}")
    
    def _camera_callback(self, msg):
        """Callback for camera image messages."""
        try:
            # Store the latest camera message
            self.latest_camera_image = msg
            # Log occasionally to confirm messages are being received
            if not hasattr(self, '_camera_callback_count'):
                self._camera_callback_count = 0
            self._camera_callback_count += 1
            
            # Log first few messages and then occasionally to confirm messages are being received
            if self._camera_callback_count <= 3 or self._camera_callback_count % 20 == 0:
                print(f"[CAMERA] Received image #{self._camera_callback_count}: {msg.width}x{msg.height}, encoding={msg.encoding}")
            
        except Exception as e:
            print(f"[CAMERA] Error in camera callback: {e}")
    
    def _setup_llm_worker(self):
        """Setup background thread for LLM API calls."""
        self.llm_thread = threading.Thread(target=self._llm_worker, daemon=True)
        self.llm_thread.start()
    
    def _llm_worker(self):
        """Background worker thread for processing LLM requests."""
        while self._llm_worker_running:
            try:
                request = self.llm_request_queue.get(timeout=1.0)
                if request is None:  # Shutdown signal
                    break
                
                if not self._llm_worker_running:  # Check again after getting request
                    break
                    
                response = self._call_llm(request)
                if self._llm_worker_running:  # Only put response if still running
                    self.llm_response_queue.put(response)
                
            except Empty:
                continue
            except Exception as e:
                if self._llm_worker_running:  # Only log if still running
                    print(f"LLM worker error: {e}")
        
        print("[LLM] Worker thread stopped")
    
    def _encode_camera_image(self) -> Optional[str]:
        """Encode current camera image to base64 for LLM input using ROS topic data or BeamNG sensor."""
        try:
            if not self.use_camera_input:
                return None
            
            # Try ROS topic approach first
            if self.latest_camera_image is not None:
                return self._encode_from_ros_image()
            
            # Fallback to BeamNG sensor approach
            print(f"[CAMERA] No ROS image received, trying BeamNG sensor fallback")
            return self._encode_from_beamng_sensor()
            
        except Exception as e:
            print(f"[CAMERA] Error encoding camera image: {e}")
            return None
    
    def _encode_from_ros_image(self) -> Optional[str]:
        """Encode camera image from ROS topic data."""
        try:
            # Convert ROS Image message to PIL Image
            from cv_bridge import CvBridge
            import cv2
            
            bridge = CvBridge()
            
            # Convert ROS Image message to OpenCV image
            cv_image = bridge.imgmsg_to_cv2(self.latest_camera_image, desired_encoding='bgr8')
            print(f"[CAMERA] Converted ROS image to OpenCV: {cv_image.shape}")
            
            # Convert BGR to RGB for PIL
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_img = Image.fromarray(rgb_image)
            print(f"[CAMERA] Converted to PIL image: {pil_img.size}")
            
            # Resize if requested
            if self.resize_image:
                pil_img = pil_img.resize(self.resize_image, Image.LANCZOS)
                print(f"[CAMERA] Resized image to: {pil_img.size}")
            
            # Convert to base64
            buffer = io.BytesIO()
            pil_img.save(buffer, format='JPEG', quality=self.image_quality)
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            print(f"[CAMERA] Successfully encoded ROS image to base64 ({len(img_base64)} chars)")
            return img_base64
            
        except Exception as e:
            print(f"[CAMERA] Error encoding ROS image: {e}")
            return None
    
    def _encode_from_beamng_sensor(self) -> Optional[str]:
        """Encode camera image from BeamNG vehicle sensor."""
        try:
            if not self.vehicle:
                print(f"[CAMERA] No vehicle available for BeamNG sensor access")
                return None
            
            # Try to get camera sensor from vehicle
            camera_sensor_name = 'front_cam'  # This should match the sensor name in the scenario
            
            # Access the sensor data directly
            state_data = self.vehicle.sensors.poll()
            
            if state_data is None:
                print(f"[CAMERA] Vehicle sensor polling returned None")
                return None
            
            if camera_sensor_name not in state_data:
                print(f"[CAMERA] Camera sensor '{camera_sensor_name}' not found in sensor data. Available: {list(state_data.keys())}")
                return None
            
            camera_data = state_data[camera_sensor_name]
            
            if camera_data is None or 'colour' not in camera_data:
                print(f"[CAMERA] No colour data in camera sensor")
                return None
            
            # Get the image data
            colour_data = camera_data['colour']
            
            # Convert numpy array to PIL Image
            import numpy as np
            
            if isinstance(colour_data, np.ndarray):
                # Assume it's in RGB format from BeamNG
                pil_img = Image.fromarray(colour_data.astype('uint8'))
                print(f"[CAMERA] Created PIL image from BeamNG sensor: {pil_img.size}")
                
                # Resize if requested
                if self.resize_image:
                    pil_img = pil_img.resize(self.resize_image, Image.LANCZOS)
                    print(f"[CAMERA] Resized image to: {pil_img.size}")
                
                # Convert to base64
                buffer = io.BytesIO()
                pil_img.save(buffer, format='JPEG', quality=self.image_quality)
                img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                print(f"[CAMERA] Successfully encoded BeamNG image to base64 ({len(img_base64)} chars)")
                return img_base64
            else:
                print(f"[CAMERA] Unexpected camera data type: {type(colour_data)}")
                return None
            
        except Exception as e:
            print(f"[CAMERA] Error encoding BeamNG camera image: {e}")
            import traceback
            print(f"[CAMERA] Traceback: {traceback.format_exc()}")
            return None
    
    
    def _call_llm(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Make API call to LLM service."""
        try:
            # Only support OpenRouter provider
            provider = self.llm_config.get("provider", "openrouter")
            
            if provider == "openrouter":
                return self._call_openrouter(context)
            else:
                print(f"Unsupported LLM provider: {provider}. Only 'openrouter' is supported.")
                return None
                
        except Exception as e:
            print(f"LLM API call failed: {e}")
            return None
    
    
    def _call_openrouter(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Call OpenRouter API for prompt generation."""
        try:
            import openai
            
            client = openai.OpenAI(
                api_key=self.llm_config.get("api_key"),
                base_url="https://openrouter.ai/api/v1"
            )
            
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_context_prompt(context)
            
            # OpenRouter requires specific headers
            extra_headers = {
                "HTTP-Referer": self.llm_config.get("app_name", "beamng-ros2-integration"),
                "X-Title": self.llm_config.get("app_name", "BeamNG VLA Training")
            }
            
            # Build messages with vision support (OpenRouter supports OpenAI-style vision)
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
            
            response = client.chat.completions.create(
                model=self.llm_config.get("model", "anthropic/claude-3-sonnet" if "camera_image" in context else "anthropic/claude-3-haiku"),
                messages=messages,
                temperature=self.llm_config.get("temperature", 0.7),
                max_tokens=self.llm_config.get("max_tokens", 200),
                extra_headers=extra_headers
            )
            
            return self._parse_llm_response(response.choices[0].message.content)
            
        except Exception as e:
            print(f"OpenRouter API error: {e}")
            return None
    
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for LLM."""
        return f"""You are an expert driving instructor providing contextual guidance for autonomous vehicle training.

Generate concise, specific driving prompts based on the current vehicle state and scenario context.

Respond with a JSON object containing:
{{
    "task_description": "Brief description of current driving task",
    "current_objective": "Immediate driving objective", 
    "driving_instructions": ["instruction1", "instruction2", "instruction3"],
    "required_skills": ["skill1", "skill2"]
}}

Focus on:
- Current driving conditions and vehicle state
- Specific actions the driver should take
- Safety considerations
- Terrain-appropriate guidance
- Clear, actionable instructions

Scenario context: {self.scenario_name} on {self.level_name}
Road type: {self.road_type}
Weather: {self.weather_conditions}
Time: {self.time_of_day}"""
    
    def _build_context_prompt(self, context: Dict[str, Any]) -> str:
        """Build context-specific prompt for LLM."""
        speed = context.get("speed", 0)
        position = context.get("position", [0, 0, 0])
        moving = context.get("moving", False)
        
        prompt = f"""Current vehicle state:
- Speed: {speed:.1f} m/s
- Position: {position}
- Moving: {moving}
- Terrain: {self.road_type}
"""
        
        if self.use_camera_input:
            prompt += "\n- Front camera view: Provided as image input"
        
        prompt += "\n\nGenerate appropriate driving guidance for this situation. Be specific about what the driver should do right now."
        if self.use_camera_input:
            prompt += " Consider what you can see in the front camera view."
            
        return prompt
    
    def _parse_llm_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Parse LLM response into structured format."""
        try:
            # Try to extract JSON from response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                return json.loads(json_str)
            else:
                # Fallback: create structured response from text
                lines = response_text.strip().split('\n')
                return {
                    "task_description": lines[0] if lines else self.base_task_description,
                    "current_objective": lines[1] if len(lines) > 1 else self.base_objective,
                    "driving_instructions": lines[2:5] if len(lines) > 2 else self.base_instructions,
                    "required_skills": ["general_driving"]
                }
                
        except Exception as e:
            print(f"Failed to parse LLM response: {e}")
            return None
    
    def _generate_dynamic_context(self) -> Dict[str, Any]:
        """Generate dynamic context based on current vehicle state."""
        print(f"[DEBUG] _generate_dynamic_context called: enable_dynamic_prompts={self.enable_dynamic_prompts}, vehicle={self.vehicle is not None}")
        if not self.enable_dynamic_prompts or not self.vehicle:
            print(f"[DEBUG] Early return from _generate_dynamic_context")
            return {}
        
        try:
            # Access state sensor data safely
            try:
                state_sensor_data = self.vehicle.sensors['state'].data
                print(f"[DEBUG] State sensor data keys: {list(state_sensor_data.keys()) if state_sensor_data else 'None'}")
            except (KeyError, AttributeError) as e:
                print(f"[DEBUG] Cannot access 'state' sensor: {e}")
                # Return basic static context when state sensor isn't available
                return {
                    'current_objective': "Initialize and prepare for driving",
                    'driving_instructions': self.base_instructions + ["Vehicle starting up", "Sensors initializing"],
                    'task_description': f"{self.base_task_description} - Vehicle initialization phase"
                }
            
            if not state_sensor_data or 'vel' not in state_sensor_data or 'pos' not in state_sensor_data:
                print(f"[DEBUG] Missing vel/pos in state sensor data - using fallback static context")
                return {
                    'current_objective': "Initialize and prepare for driving",
                    'driving_instructions': self.base_instructions + ["Vehicle starting up", "Sensors initializing"],
                    'task_description': f"{self.base_task_description} - Vehicle initialization phase"
                }
                
            current_speed = (state_sensor_data['vel'][0]**2 + state_sensor_data['vel'][1]**2 + state_sensor_data['vel'][2]**2)**0.5
            current_pos = state_sensor_data['pos']
            
            # Check if vehicle is moving
            distance_moved = 0.0
            if self.last_position != [0.0, 0.0, 0.0]:
                distance_moved = ((current_pos[0] - self.last_position[0])**2 + 
                                (current_pos[1] - self.last_position[1])**2)**0.5
            
            moving = distance_moved > 1.0 or current_speed > 1.0
            
            # Prepare context for LLM or rule-based generation
            vehicle_context = {
                "speed": current_speed,
                "position": current_pos,
                "moving": moving,
                "distance_moved": distance_moved
            }
            
            # Use LLM if enabled, otherwise use rule-based approach
            if self.use_llm_prompts:
                print(f"[DEBUG] Calling _get_llm_context with vehicle_context: {vehicle_context}")
                llm_result = self._get_llm_context(vehicle_context)
                print(f"[DEBUG] _get_llm_context returned: {llm_result}")
                return llm_result
            else:
                print(f"[DEBUG] Using rule-based context")
                return self._get_rule_based_context(vehicle_context, current_speed, current_pos, distance_moved)
            
        except Exception as e:
            import traceback
            print(f"Error generating dynamic context: {e}")
            print(f"Exception type: {type(e)}")
            print(f"Exception details: {repr(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            return {}
    
    def _should_generate_new_prompt(self, current_time: float, vehicle_context: Dict[str, Any]) -> bool:
        """Determine if we should generate a new LLM prompt based on timing and state changes."""
        
        # Check time interval
        time_since_last = current_time - self.last_prompt_time
        print(f"[DEBUG] _should_generate_new_prompt: current_time={current_time}, last_prompt_time={self.last_prompt_time}, time_since_last={time_since_last}, prompt_interval={self.prompt_interval}")
        if time_since_last >= self.prompt_interval:
            print(f"[DEBUG] Time trigger: {time_since_last} >= {self.prompt_interval}")
            return True
        
        # Check for significant state changes if enabled
        if self.prompt_on_state_change and self.last_vehicle_state:
            speed_change = abs(vehicle_context["speed"] - self.last_vehicle_state.get("speed", 0))
            position_change = vehicle_context["distance_moved"]
            
            # Trigger on significant speed change (>5 m/s) or large position change
            if speed_change > 5.0 or position_change > 50.0:
                print(f"[PROMPT] State change trigger: speed_change={speed_change:.1f}m/s, position_change={position_change:.1f}m")
                return True
        
        print(f"[DEBUG] No trigger conditions met")
        return False
    
    def _get_llm_context(self, vehicle_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get context from LLM generation with timing controls."""
        try:
            print(f"[DEBUG] _get_llm_context started - always generating new LLM request since timing is controlled by outer method")
            
            # Always generate new prompt since timing is controlled by get_data()
            # Add camera image to context if enabled
            if self.use_camera_input:
                camera_image = self._encode_camera_image()
                if camera_image:
                    vehicle_context["camera_image"] = camera_image
                    print(f"[LLM] Added camera image to context ({len(camera_image)} chars)")
            
            # Submit request to LLM worker (non-blocking)
            self.llm_request_queue.put(vehicle_context)
            self.last_vehicle_state = vehicle_context.copy()
            print(f"[LLM] Requesting new prompt")
            
            # Try to get recent LLM response
            try:
                llm_response = self.llm_response_queue.get_nowait()
                if llm_response:
                    self.current_llm_prompt = llm_response
                    print(f"[LLM] Received new prompt: {llm_response.get('task_description', 'N/A')[:50]}...")
            except Empty:
                pass  # No new response available
            
            # Use latest LLM response or fallback
            if self.current_llm_prompt:
                print(f"[DEBUG] Using cached LLM prompt: {self.current_llm_prompt}")
                return self.current_llm_prompt
            else:
                print(f"[DEBUG] No LLM prompt available, falling back to rule-based")
                # Fallback to rule-based while waiting for LLM
                return self._get_rule_based_context(
                    vehicle_context, 
                    vehicle_context["speed"], 
                    vehicle_context["position"], 
                    vehicle_context["distance_moved"]
                )
                
        except Exception as e:
            print(f"LLM context generation error: {e}")
            return {}
    
    def _get_rule_based_context(self, vehicle_context: Dict[str, Any], current_speed: float, 
                               current_pos: list, distance_moved: float) -> Dict[str, Any]:
        """Get context using rule-based approach (fallback or primary)."""
        context = {}
        
        # Speed-based context
        if current_speed < 5:  # Very slow/stopped
            context['current_objective'] = "Start moving or navigate carefully at low speed"
            context['driving_instructions'] = self.base_instructions + ["Accelerate gradually", "Check surroundings"]
        elif current_speed > 30:  # High speed
            context['current_objective'] = "Maintain high speed safely"
            context['driving_instructions'] = self.base_instructions + ["Maintain safe distance", "Be ready to brake"]
        else:
            context['current_objective'] = self.base_objective
            context['driving_instructions'] = self.base_instructions
        
        # Position change context
        if self.last_position != [0.0, 0.0, 0.0]:
            if distance_moved < 1.0 and current_speed < 1.0:  # Not moving much
                context['task_description'] = f"{self.base_task_description} - Currently stationary or moving slowly"
            elif distance_moved > 100:  # Moved significantly
                context['task_description'] = f"{self.base_task_description} - Actively navigating terrain"
            else:
                context['task_description'] = self.base_task_description
        else:
            context['task_description'] = self.base_task_description
        
        # Update state tracking
        self.last_speed = current_speed
        self.last_position = current_pos
        
        return context
    
    def get_data(self, time: Time) -> msgs.DrivingPrompt:
        """
        Generate the driving prompt message with current scenario context.
        
        Args:
            time: ROS2 timestamp for the message header
            
        Returns:
            DrivingPrompt message with structured scenario information
        """
        # Check if publisher has been stopped
        if self._publisher_stopped:
            return None
            
        # Apply timing logic for all prompts (both LLM and static)
        import time as system_time
        current_time = system_time.time()  # Use consistent system time
        time_since_last = current_time - self.last_prompt_time
        
        if time_since_last < self.prompt_interval:
            return None
        
        print(f"[PROMPT] Time trigger: {time_since_last:.1f}s >= {self.prompt_interval}s - generating prompt")
        self.last_prompt_time = current_time
        
        # Get dynamic context if enabled
        print(f"[DEBUG] About to call _generate_dynamic_context(), use_llm_prompts={self.use_llm_prompts}")
        dynamic_context = self._generate_dynamic_context()
        print(f"[DEBUG] Dynamic context result: {dynamic_context}")
        
        # Use dynamic context or fall back to base values
        task_description = dynamic_context.get('task_description', self.base_task_description)
        current_objective = dynamic_context.get('current_objective', self.base_objective)
        driving_instructions = dynamic_context.get('driving_instructions', self.base_instructions)
        
        msg = msgs.DrivingPrompt(
            header=self._make_header(time),
            task_description=task_description,
            scenario_name=self.scenario_name,
            level_name=self.level_name,
            weather_conditions=self.weather_conditions,
            time_of_day=self.time_of_day,
            current_objective=current_objective,
            traffic_density=self.traffic_density,
            road_type=self.road_type,
            visibility_conditions=self.visibility_conditions,
            driving_instructions=driving_instructions,
            target_location=Point(
                x=float(self.target_location[0]),
                y=float(self.target_location[1]),
                z=float(self.target_location[2])
            ),
            difficulty_level=self.difficulty_level,
            required_skills=self.required_skills
        )
        return msg
    
    def publish(self, time: Time):
        """Publish the driving prompt message."""
        if self._publisher_stopped:
            return
            
        data = self.get_data(time)
        if data is not None:
            print(f"[PROMPT] Publishing prompt: '{data.task_description[:60]}...' (objective: {data.current_objective})")
            self._publisher.publish(data)
    
    def remove(self):
        """Clean up resources when the sensor is removed."""
        print("[PROMPT] Stopping prompt publisher...")
        
        # Set stop flag to prevent further publishing
        self._publisher_stopped = True
        
        # Stop the LLM worker thread if running
        if hasattr(self, '_llm_worker_running'):
            self._llm_worker_running = False
            
        # Send shutdown signal to queue
        if hasattr(self, 'llm_request_queue'):
            self.llm_request_queue.put(None)
            
        # Wait for thread to finish
        if hasattr(self, 'llm_thread') and self.llm_thread and self.llm_thread.is_alive():
            self.llm_thread.join(timeout=2.0)
            if self.llm_thread.is_alive():
                print("[PROMPT] Warning: LLM thread did not stop cleanly")
            else:
                print("[PROMPT] LLM thread stopped cleanly")
        
        print("[PROMPT] Prompt publisher stopped")


def create_adaptive_prompt_config(scenario_spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create an adaptive prompt configuration based on the scenario specification.
    This function analyzes the scenario and generates appropriate prompts.
    
    Args:
        scenario_spec: The scenario specification dictionary
        
    Returns:
        Configuration dictionary for the DrivingPromptPublisher
    """
    # Check if external prompt config file is specified
    if "prompt_config_file" in scenario_spec:
        config_file_path = scenario_spec["prompt_config_file"]
        try:
            import json
            from pathlib import Path
            
            # Handle relative paths from the scenario file location
            if not Path(config_file_path).is_absolute():
                # First try relative to the source package root (development)
                source_package_root = Path(__file__).parent.parent.parent.parent
                config_file_path_src = source_package_root / config_file_path
                
                if config_file_path_src.exists():
                    config_file_path = config_file_path_src
                else:
                    # Fallback to installed package location
                    package_root = Path(__file__).parent.parent.parent
                    config_file_path = package_root / config_file_path
            
            with open(config_file_path, 'r') as f:
                external_config = json.load(f)
                
            # If the config has a nested structure (like vision_prompt_config.json)
            if "prompt_publisher_config" in external_config:
                config = external_config["prompt_publisher_config"]
            else:
                config = external_config
                
            print(f"Loaded external prompt config from: {config_file_path}")
            return config
            
        except FileNotFoundError:
            print(f"Warning: Prompt config file not found: {config_file_path}")
            print("Falling back to auto-generated config")
        except json.JSONDecodeError as e:
            print(f"Warning: Invalid JSON in prompt config file: {e}")
            print("Falling back to auto-generated config")
        except Exception as e:
            print(f"Warning: Error loading prompt config: {e}")
            print("Falling back to auto-generated config")
    level_name = scenario_spec.get("level", "unknown")
    scenario_name = scenario_spec.get("name", "unknown")
    
    # Map levels to road types and contexts
    level_contexts = {
        "johnson_valley": {
            "road_type": "off_road",
            "terrain": "desert",
            "difficulty_level": "high",
            "required_skills": ["off_road_driving", "terrain_navigation", "vehicle_control"]
        },
        "italy": {
            "road_type": "mountain_road",
            "terrain": "mountainous",
            "difficulty_level": "medium",
            "required_skills": ["curve_handling", "elevation_changes", "scenic_driving"]
        },
        "west_coast_usa": {
            "road_type": "highway",
            "terrain": "coastal",
            "difficulty_level": "medium",
            "required_skills": ["highway_driving", "lane_changing", "speed_control"]
        }
    }
    
    context = level_contexts.get(level_name, {
        "road_type": "mixed",
        "terrain": "varied",
        "difficulty_level": "medium",
        "required_skills": ["general_driving"]
    })
    
    # Extract weather and time information
    weather_conditions = "clear"
    time_of_day = "day"
    
    if "weather_presets" in scenario_spec:
        weather_conditions = scenario_spec["weather_presets"]
    
    if "time_of_day" in scenario_spec:
        time_of_day = str(scenario_spec["time_of_day"])
    
    # Generate task description based on scenario
    task_descriptions = {
        "johnson_valley": "Navigate through challenging off-road terrain in Johnson Valley",
        "italy": "Drive through scenic Italian mountain roads with careful attention to curves",
        "west_coast_usa": "Drive along the scenic West Coast highway maintaining safe speeds"
    }
    
    config = {
        "task_description": task_descriptions.get(
            level_name, 
            f"Drive safely in the {scenario_name} scenario"
        ),
        "scenario_name": scenario_name,
        "level_name": level_name,
        "weather_conditions": weather_conditions,
        "time_of_day": time_of_day,
        "current_objective": "Complete the driving scenario safely and efficiently",
        "traffic_density": "low",  # BeamNG scenarios typically have low traffic
        "road_type": context["road_type"],
        "visibility_conditions": "good" if weather_conditions == "clear" else "reduced",
        "driving_instructions": [
            "Maintain safe following distance",
            "Observe speed limits",
            "Stay in designated lanes when possible",
            f"Adapt driving style for {context['terrain']} terrain"
        ],
        "target_location": [0.0, 0.0, 0.0],  # Can be updated based on specific waypoints
        "difficulty_level": context["difficulty_level"],
        "required_skills": context["required_skills"],
        "enable_dynamic_prompts": True,  # Enable context-aware prompts by default
        "use_camera_input": False,  # Set to True to enable camera vision input to LLMs
        "camera_name": "default",  # Name of camera sensor to use
        "resize_image": [256, 256],  # Resize images for faster processing
        "image_quality": 85  # JPEG quality for image compression
    }
    
    return config