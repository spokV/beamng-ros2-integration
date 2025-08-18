#!/usr/bin/env python3
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
    def __init__(self, prompt_interval=10.0):
        super().__init__('standalone_vision_llm')
        
        # Configuration
        self.prompt_interval = prompt_interval  # Generate prompt every N seconds
        self.llm_config = {
            "provider": "openrouter",
            "api_key": "",
            "model": "anthropic/claude-3-haiku",
            "temperature": 0.7,
            "max_tokens": 300
        }
        
        # State tracking
        self.latest_camera_image = None
        self.latest_vehicle_state = None
        self.camera_callback_count = 0
        self.state_callback_count = 0
        self.last_prompt_time = 0.0
        
        # LLM processing
        self.llm_request_queue = Queue()
        self.llm_response_queue = Queue()
        self.current_llm_prompt = None
        self._llm_worker_running = True
        self.llm_thread = threading.Thread(target=self._llm_worker, daemon=True)
        self.llm_thread.start()
        
        self._setup_subscribers()
        self._setup_publisher()
        self._setup_timer()
        
        self.get_logger().info(f"Standalone Vision-LLM system started! (Prompt interval: {self.prompt_interval}s)")
    
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
            "/standalone_vision_llm/driving_prompt",
            1
        )
        self.get_logger().debug("Created prompt publisher at /standalone_vision_llm/driving_prompt")
    
    def _setup_timer(self):
        """Setup timer for periodic prompt generation."""
        self.prompt_timer = self.create_timer(self.prompt_interval, self._generate_prompt)
        self.get_logger().debug(f"Created prompt generation timer (every {self.prompt_interval}s)")
    
    def _camera_callback(self, msg):
        """Callback for camera image messages."""
        self.latest_camera_image = msg
        self.camera_callback_count += 1
        
        if self.camera_callback_count <= 3 or self.camera_callback_count % 20 == 0:
            self.get_logger().debug(
                f"[CAMERA] Received image #{self.camera_callback_count}: "
                f"{msg.width}x{msg.height}, encoding={msg.encoding}"
            )
    
    def _state_callback(self, msg):
        """Callback for vehicle state messages."""
        self.latest_vehicle_state = msg
        self.state_callback_count += 1
        
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
            current_time = time.time()
            
            # Check if we have the required data
            if self.latest_camera_image is None:
                self.get_logger().debug("No camera image available - skipping prompt generation")
                return
            
            if self.latest_vehicle_state is None:
                self.get_logger().debug("No vehicle state available - skipping prompt generation")
                return
            
            # Prepare context for LLM
            context = self._prepare_llm_context()
            
            # Submit to LLM worker (non-blocking)
            self.llm_request_queue.put(context)
            
            # Try to get recent LLM response
            try:
                llm_response = self.llm_response_queue.get_nowait()
                if llm_response:
                    self.current_llm_prompt = llm_response
                    
                    # Log the complete LLM response
                    self.get_logger().info("="*80)
                    self.get_logger().info("[LLM] COMPLETE PROMPT RECEIVED:")
                    self.get_logger().info("="*80)
                    self.get_logger().info(f"Task Description: {llm_response.get('task_description', 'N/A')}")
                    self.get_logger().info(f"Current Objective: {llm_response.get('current_objective', 'N/A')}")
                    self.get_logger().info(f"Driving Instructions:")
                    for i, instruction in enumerate(llm_response.get('driving_instructions', []), 1):
                        self.get_logger().info(f"  {i}. {instruction}")
                    self.get_logger().info(f"Required Skills: {llm_response.get('required_skills', [])}")
                    self.get_logger().info("="*80)
            except Empty:
                pass  # No new response available
            
            # Publish prompt (use latest available or fallback)
            self._publish_prompt()
            
        except Exception as e:
            self.get_logger().error(f"Error generating prompt: {e}")
    
    def _prepare_llm_context(self) -> Dict[str, Any]:
        """Prepare context for LLM including camera image and vehicle state."""
        # Vehicle state info
        state = self.latest_vehicle_state
        speed = np.sqrt(state.velocity.x**2 + state.velocity.y**2 + state.velocity.z**2)
        
        context = {
            "speed": speed,
            "position": [state.position.x, state.position.y, state.position.z],
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
                pil_img = PILImage.fromarray(img_array, 'RGB')
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
        while self._llm_worker_running:
            try:
                context = self.llm_request_queue.get(timeout=1.0)
                if context is None:  # Shutdown signal
                    break
                
                response = self._call_llm(context)
                if response and self._llm_worker_running:
                    self.llm_response_queue.put(response)
                
            except Empty:
                continue
            except Exception as e:
                if self._llm_worker_running:
                    self.get_logger().error(f"LLM worker error: {e}")
    
    def _call_llm(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Call LLM API with vision and state context."""
        try:
            import openai
            
            client = openai.OpenAI(
                api_key=self.llm_config["api_key"],
                base_url="https://openrouter.ai/api/v1"
            )
            
            system_prompt = """You are an expert driving instructor providing contextual guidance for autonomous vehicle training.

Generate concise, specific driving prompts based on the current vehicle state and camera view.

Respond with a JSON object containing:
{
    "task_description": "Brief description of current driving task",
    "current_objective": "Immediate driving objective", 
    "driving_instructions": ["instruction1", "instruction2", "instruction3"],
    "required_skills": ["skill1", "skill2"]
}

Focus on:
- Current driving conditions and vehicle state
- What you can see in the camera view
- Specific actions the driver should take
- Safety considerations
- Terrain-appropriate guidance"""
            
            user_prompt = f"""Current vehicle state:
- Speed: {context['speed']:.1f} m/s
- Position: {context['position']}
- Moving: {context['moving']}

Camera view: Provided as image input

Generate appropriate driving guidance for this situation. Consider what you can see in the front camera view and provide specific instructions."""
            
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
                model=self.llm_config["model"],
                messages=messages,
                temperature=self.llm_config["temperature"],
                max_tokens=self.llm_config["max_tokens"],
                extra_headers={
                    "HTTP-Referer": "beamng-ros2-integration",
                    "X-Title": "BeamNG VLA Training"
                }
            )
            
            return self._parse_llm_response(response.choices[0].message.content)
            
        except Exception as e:
            self.get_logger().error(f"LLM API error: {e}")
            return None
    
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
            # Use latest LLM response or create fallback
            if self.current_llm_prompt:
                prompt_data = self.current_llm_prompt
            else:
                # Fallback prompt based on vehicle state
                state = self.latest_vehicle_state
                speed = np.sqrt(state.velocity.x**2 + state.velocity.y**2 + state.velocity.z**2)
                
                if speed < 5:
                    prompt_data = {
                        "task_description": "Navigate at low speed with visual guidance",
                        "current_objective": "Accelerate gradually and observe surroundings", 
                        "driving_instructions": ["Check camera view", "Accelerate smoothly", "Maintain control"],
                        "required_skills": ["low_speed_control", "visual_navigation"]
                    }
                else:
                    prompt_data = {
                        "task_description": "Navigate at moderate speed with visual guidance",
                        "current_objective": "Maintain safe speed and observe terrain",
                        "driving_instructions": ["Monitor camera view", "Adapt to terrain", "Maintain safe distance"],
                        "required_skills": ["speed_control", "visual_navigation"]
                    }
            
            # Calculate current vehicle speed
            state = self.latest_vehicle_state
            current_speed = np.sqrt(state.velocity.x**2 + state.velocity.y**2 + state.velocity.z**2)
            
            # Create ROS message
            msg = DrivingPrompt()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'ego'
            
            msg.task_description = prompt_data.get('task_description', 'Navigate safely')
            msg.scenario_name = 'johnson_valley_vla_training'
            msg.level_name = 'johnson_valley'
            msg.weather_conditions = 'clear'
            msg.time_of_day = 'day'
            msg.current_objective = prompt_data.get('current_objective', 'Drive safely')
            msg.traffic_density = 'low'
            msg.road_type = 'off_road'
            msg.visibility_conditions = 'good'
            msg.driving_instructions = prompt_data.get('driving_instructions', ['Drive carefully'])
            msg.target_location = Point(x=0.0, y=0.0, z=0.0)
            msg.difficulty_level = 'medium'
            msg.required_skills = prompt_data.get('required_skills', ['general_driving'])
            
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
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging (shows camera/state details)')
    
    parsed_args = parser.parse_args(args)
    
    rclpy.init()
    
    node = StandaloneVisionLLM(prompt_interval=parsed_args.interval)
    # Update model if specified
    node.llm_config["model"] = parsed_args.model
    
    # Set log level based on debug flag
    if parsed_args.debug:
        node.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)
    else:
        node.get_logger().set_level(rclpy.logging.LoggingSeverity.INFO)
    
    node.get_logger().info(f"Starting with prompt interval: {parsed_args.interval}s, model: {parsed_args.model}")
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down standalone vision-LLM system...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()