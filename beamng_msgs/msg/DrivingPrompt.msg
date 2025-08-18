std_msgs/Header header

# High-level driving task description
string task_description

# Scenario context information  
string scenario_name
string level_name
string weather_conditions
string time_of_day

# Current driving objective
string current_objective

# Traffic/environment context
string traffic_density
string road_type
string visibility_conditions

# Specific instructions or constraints
string[] driving_instructions

# Goal location or target (if applicable)
geometry_msgs/Point target_location

# Additional metadata for VLA training
string difficulty_level
string[] required_skills