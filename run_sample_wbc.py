from leapp.inference_manager import InferenceManager
from examples.compass_native_python import CompassNavigationModel, create_test_data

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


def main():
    compass_model = CompassNavigationModel(
        mobility_model_path="examples/models/digit_mobility.jit",
        device="cuda"
    )
    image, odom, transform, goal_pose, route_transform = create_test_data(
        device=compass_model.device)
    inputs = {
        'compass_goal_checker/goal': goal_pose.clone(),
        'compass_image_processor/raw_image': image.clone(),
        'compass_odometry_processor/odom_msg': odom.clone(),
        'compass_odometry_processor/transform': transform.clone(),
        'compass_route_calculator/goal_pose': goal_pose.clone(),
        'compass_route_calculator/transform': route_transform.clone(),
    }

    for i in range(1):
        # Test hypothesis: exported model has buffers from a previous run
        # Run native model TWICE on same instance and compare 2nd run with exported
        compass_model = CompassNavigationModel(
            mobility_model_path="examples/models/digit_mobility.jit",
            device="cuda"
        )
        print("Native model - 1st run (fresh state):")
        final_commands_run1 = compass_model.run_navigation_pipeline(
            image, odom, goal_pose, transform, route_transform)
        print(f"  Output: {final_commands_run1}")

        print("\nNative model - 2nd run (with state from 1st run):")
        final_commands_run2 = compass_model.run_navigation_pipeline(
            image, odom, goal_pose, transform, route_transform)
        print(f"  Output: {final_commands_run2}")
        print(f"  action buffer: {compass_model.action}")
        print(f"  history buffer mean: {compass_model.history.mean():.6f}")
        print(f"  sample buffer mean: {compass_model.sample.mean():.6f}")

        inference_manager = InferenceManager(
            "sample_compass_navigation_pipeline/sample_compass_navigation_pipeline.yaml")

        # Initialize feedback state values to match native model's initial state
        inference_manager.set_input_value(
            "compass_odometry_processor", "prev_transform",
            torch.tensor([0.05, 0.02, 0.01, 0.99, 0.0, 0.01, 0.0, 999.0, 0.0],
                         dtype=DTYPE, device=DEVICE))
        inference_manager.set_input_value(
            "compass_odometry_processor", "ego_speed",
            torch.zeros(1, dtype=DTYPE, device=DEVICE))
        inference_manager.set_input_value(
            "compass_odometry_processor", "position_2d",
            torch.zeros(3, dtype=DTYPE, device=DEVICE))

        # Debug: Test odometry processor with SAME inputs as native model
        print("\n=== Debug: Testing compass_odometry_processor with identical inputs ===")

        # These are the exact inputs the native model uses on first call
        native_compass = CompassNavigationModel(
            mobility_model_path="examples/models/digit_mobility.jit",
            device="cuda"
        )
        # [0.05, 0.02, 0.01, 0.99, 0.0, 0.01, 0.0, 999.0, 0.0]
        native_prev_transform = native_compass.transform.clone()
        native_ego_speed = native_compass.speed.clone()  # zeros(1)
        native_position_2d = native_compass.position_2d.clone()  # zeros(3)

        print(f"Input odom_msg: {odom}")
        print(f"Input transform: {transform}")
        print(f"Input prev_transform: {native_prev_transform}")
        print(f"Input ego_speed: {native_ego_speed}")
        print(f"Input position_2d: {native_position_2d}")

        # Run native model
        native_new_transform, native_speed, native_pos2d = native_compass.odom_processor.process(
            odom, transform, native_prev_transform, native_ego_speed, native_position_2d)
        print(f"\nNative outputs:")
        print(f"  new_transform: {native_new_transform}")
        print(f"  speed: {native_speed}")
        print(f"  position_2d: {native_pos2d}")

        # Run exported model with SAME inputs
        exported_odom = inference_manager.nodes["compass_odometry_processor"]
        exported_outputs = exported_odom(
            odom.clone(), transform.clone(), native_prev_transform.clone(),
            native_ego_speed.clone(), native_position_2d.clone())
        print(f"\nExported outputs:")
        print(f"  new_transform: {exported_outputs[0]}")
        print(f"  speed: {exported_outputs[1]}")
        print(f"  position_2d: {exported_outputs[2]}")

        # Check if they match
        if torch.allclose(native_new_transform, exported_outputs[0], atol=1e-5):
            print("\n✓ new_transform MATCHES")
        else:
            print(
                f"\n✗ new_transform DIFFERS by {(native_new_transform - exported_outputs[0]).abs().max().item()}")

        # Run the full policy with verbose mode
        print("\n=== Running exported pipeline ===")
        outputs = inference_manager.run_policy(inputs, verbose=False)
        print(f"Exported output: {outputs}")

        # Compare
        exported_cmd = outputs['post_process_commands/cmd']
        print(f"\n=== COMPARISON ===")
        print(f"Native 1st run: {final_commands_run1}")
        print(f"Native 2nd run: {final_commands_run2}")
        print(f"Exported:       {exported_cmd}")

        if torch.allclose(final_commands_run1, exported_cmd, atol=1e-4):
            print("\n✓ Exported matches Native 1st run")
        elif torch.allclose(final_commands_run2, exported_cmd, atol=1e-4):
            print(
                "\n✓ Exported matches Native 2nd run -> BUFFERS WERE CAPTURED WITH STATE!")
        else:
            print("\n✗ Exported matches neither run")


if __name__ == "__main__":
    main()
