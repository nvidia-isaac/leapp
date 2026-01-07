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

    for i in range(5):
        final_commands = compass_model.run_navigation_pipeline(
                image, odom, goal_pose, transform, route_transform)
        print(final_commands)
    compass_policy = InferenceManager(
        "sample_compass_navigation_pipeline/sample_compass_navigation_pipeline.yaml", verbose=True)

    for i in range(5):
        outputs = compass_policy.run_policy(inputs)
        print(outputs)
    # inputs = {
    #     'compass_goal_checker/goal': goal_pose,
    #     'compass_image_processor/raw_image': image,
    #     'compass_odometry_processor/odom_msg': odom,
    #     'compass_odometry_processor/transform': transform,
    #     'compass_odometry_processor/prev_transform': transform.clone(),
    #     'compass_odometry_processor/ego_speed': torch.zeros(1, device=DEVICE, dtype=DTYPE),
    #     'compass_odometry_processor/position_2d': torch.zeros(3, device=DEVICE, dtype=DTYPE),
    #     'compass_route_calculator/goal_pose': goal_pose,
    #     'compass_route_calculator/transform': route_transform,
    # }

    # }

    # for i in range(10):
    #     inputs = {
    #         'concatenate_and_run_model/velocity_commands': torch.randn(3, device=DEVICE, dtype=DTYPE),
    #         'concatenate_and_run_model/joint_vel': torch.randn(19, device=DEVICE, dtype=DTYPE),
    #         'process_joint_pos/joint_pos': torch.randn(19, device=DEVICE, dtype=DTYPE),
    #         'process_odom/lin_vel_I': torch.randn(3, device=DEVICE, dtype=DTYPE),
    #         'process_odom/ang_vel_I': torch.randn(3, device=DEVICE, dtype=DTYPE),
    #         'process_odom/q_IB': torch.randn(4, device=DEVICE, dtype=DTYPE),
    #     }
    #     # Extract just the input names (part after '/') for the reference WBC
    #     inputs_for_wbc = {
    #         key.split('/')[1]: value for key, value in inputs.items()}
    #     reference_outputs, previous_actions = run_model(
    #         model, **inputs_for_wbc, previous_actions=previous_actions)

    #     outputs = wbc_policy.run_policy(inputs)

    #     # Check if outputs match reference
    #     output_tensor = outputs['post_process_actions/actions']
    #     if torch.allclose(reference_outputs, output_tensor, rtol=1e-5, atol=1e-5):
    #         print(f"Iteration {i}: PASS - outputs match reference")
    #     else:
    #         print(f"Iteration {i}: FAIL - outputs do not match")
    #         print(
    #             f"  Max difference: {(reference_outputs - output_tensor).abs().max().item()}")
    #         print(f"  Reference: {reference_outputs}")
    #         print(f"  Output:    {output_tensor}")


if __name__ == "__main__":
    main()
