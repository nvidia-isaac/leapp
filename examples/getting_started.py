import torch
from leapp import annotate

# Method node: Process and normalize sensor data
@annotate.method(export_with="torch")
def process_sensor_data(raw_readings):
    """Process raw sensor readings and normalize them."""
    processed = torch.clamp(raw_readings, min=0.0, max=1.0)
    normalized = (processed - 0.5) * 2.0
    return normalized

# Helper function for feature extraction (called within traced tensor context)
def compute_obstacle_features(sensor_data):
    """Compute obstacle-related features from sensor data."""
    obstacle_distance = torch.mean(torch.abs(sensor_data))
    obstacle_variance = torch.var(sensor_data)
    return obstacle_distance, obstacle_variance

def main():
    # Start tracing our computational graph
    annotate.start(name="sample_robot_pipeline")
    
    # Create some sample sensor data
    raw_sensor_data = torch.tensor([0.1, 0.8, 0.3, 0.9, 0.2])
    
    # ===== NODE 1: Method decorator =====
    # Process sensor data using decorated function
    clean_data = process_sensor_data(raw_sensor_data)
    
    # ===== NODE 2: Traced tensors =====
    # Extract navigation features using traced tensors
    # This allows us to trace operations across function calls
    sensor_input = annotate.input_tensors({
        'sensor_data': clean_data
    }, 'feature_extractor')
    
    # Operations are automatically traced - even through helper functions!
    obstacle_dist, obstacle_var = compute_obstacle_features(sensor_input)
    
    # Additional inline operations are also traced
    safe_speed = torch.clamp(obstacle_dist, min=0.1, max=1.0)
    confidence = 1.0 / (1.0 + obstacle_var)
    
    # Mark outputs to finalize the traced node
    annotate.output_tensors({
        'safe_speed': safe_speed,
        'confidence': confidence
    }, 'feature_extractor', export_with="torch")
    
    # ===== NODE 3: Block annotation =====
    # Control decisions using annotation block
    with annotate.block("control_decision",
                         inputs=["safe_speed", "confidence"],
                         outputs=["robot_action"],
                         export_with="torch"):
        # Combine features into final robot action
        # Action format: [forward_speed, turn_rate, caution_factor]
        forward_speed = safe_speed * confidence
        turn_rate = torch.zeros(1)
        caution_factor = 1.0 - confidence
        robot_action = torch.cat([forward_speed.unsqueeze(0), 
                                   turn_rate, 
                                   caution_factor.unsqueeze(0)])
    
    # Stop tracing and compile the graph
    annotate.stop()
    annotate.compile_graph()
    
    print(f"Raw sensor data: {raw_sensor_data}")
    print(f"Processed data: {clean_data}")
    print(f"Safe speed: {safe_speed}")
    print(f"Confidence: {confidence}")
    print(f"Robot action: {robot_action}")

if __name__ == "__main__":
    main()