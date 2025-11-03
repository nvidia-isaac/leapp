#!/usr/bin/env python3

# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
import torchvision.transforms.functional as TF
import time
import argparse
import sys
import os

from leapp import annotate


class CompassImageProcessor:
    """Processes raw camera images for the navigation model."""

    def __init__(self, target_width=960, target_height=640):
        self.target_width = target_width
        self.target_height = target_height

    @annotate.method(node_name="compass_image_processor", export_with="torch")
    def process(self, raw_image: torch.Tensor) -> torch.Tensor:
        """
        Process raw image data following the original navigator._new_image logic.

        Args:
            raw_image: Raw RGB image tensor, shape [height, width, 3], dtype uint8, range [0, 255]

        Returns:
            Processed image tensor, shape [3, target_height, target_width], dtype float32, range [0, 1]
        """
        # Convert to float for processing
        image_float = raw_image.float()

        # Resize using torchvision functional - first convert to CHW format
        image_chw = image_float.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
        resized_image = TF.resize(
            image_chw, [self.target_height, self.target_width])

        # Normalize from [0, 255] to [0, 1]
        normalized_image = resized_image / 255.0

        return normalized_image


class CompassOdometryProcessor:
    """Processes odometry data and updates robot state."""

    def __init__(self):
        pass

    def quaternion_to_matrix(self, quaternions: torch.Tensor) -> torch.Tensor:
        """Convert quaternions to rotation matrices."""
        quaternions = torch.as_tensor(quaternions)
        w, x, y, z = torch.unbind(quaternions, -1)
        two_s = 2.0 / (quaternions * quaternions).sum(-1)

        o = torch.stack([
            1 - two_s * (y * y + z * z),
            two_s * (x * y - z * w),
            two_s * (x * z + y * w),
            two_s * (x * y + z * w),
            1 - two_s * (x * x + z * z),
            two_s * (y * z - x * w),
            two_s * (x * z - y * w),
            two_s * (y * z + x * w),
            1 - two_s * (x * x + y * y),
        ], -1)
        return o.reshape(quaternions.shape[:-1] + (3, 3))

    def estimate_velocities_from_transforms(self, prev_transform: torch.Tensor,
                                            current_transform: torch.Tensor,
                                            dt_s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Estimate linear and angular velocities from transforms."""
        if dt_s.abs() < 1e-10:
            return torch.zeros(3), torch.zeros(3)

        # Extract positions and orientations
        prev_pos = prev_transform[0:3]
        curr_pos = current_transform[0:3]
        prev_quat = prev_transform[3:7]  # [x, y, z, w]
        curr_quat = current_transform[3:7]  # [x, y, z, w]

        # Convert to [w, x, y, z] format for quaternion operations (if needed for future use)
        # prev_quat_wxyz = torch.stack([prev_quat[3], prev_quat[0], prev_quat[1], prev_quat[2]])
        # curr_quat_wxyz = torch.stack([curr_quat[3], curr_quat[0], curr_quat[1], curr_quat[2]])

        # Linear velocity
        lin_vel = (curr_pos - prev_pos) / dt_s

        # Angular velocity (simplified for 2D navigation)
        # Extract yaw angles from quaternions
        prev_yaw = 2 * torch.atan2(prev_quat[2], prev_quat[3])
        curr_yaw = 2 * torch.atan2(curr_quat[2], curr_quat[3])

        # Handle angle wrapping
        dyaw = curr_yaw - prev_yaw
        if dyaw > torch.pi:
            dyaw -= 2 * torch.pi
        elif dyaw < -torch.pi:
            dyaw += 2 * torch.pi

        # Create angular velocity with consistent tensor shapes
        ang_vel = torch.tensor(
            [0.0, 0.0, (dyaw / dt_s).item()], dtype=torch.float32)

        return ang_vel, lin_vel

    @annotate.method(node_name="compass_odometry_processor", export_with="torch")
    def process(self, odom_msg: torch.Tensor, transform: torch.Tensor,
                prev_transform: torch.Tensor, ego_speed: torch.Tensor,
                position_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process odometry data and update robot state.

        Args:
            odom_msg: Odometry data [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz, timestamp_sec, timestamp_nsec]
            transform: Current transform [x, y, z, qx, qy, qz, qw, timestamp_sec, timestamp_nsec]
            prev_transform: Previous transform [x, y, z, qx, qy, qz, qw, timestamp_sec, timestamp_nsec]
            ego_speed: Current ego speed [speed]
            position_2d: Current 2D position [x, y, orientation_z]

        Returns:
            (updated_transform, updated_ego_speed, updated_position_2d)
        """
        # Create copy of current transform to return as new prev_transform
        new_transform = transform.clone()

        # Update ego speed if we have valid previous transform
        if prev_transform is not None:
            dt_sec = transform[7] - prev_transform[7]
            dt_nsec = transform[8] - prev_transform[8]
            dt_s = dt_sec + dt_nsec / 1e9

            if abs(dt_s) >= 1e-10:
                dt_tensor = torch.tensor(dt_s, dtype=torch.float32)
                _, lin_vel = self.estimate_velocities_from_transforms(
                    prev_transform, transform, dt_tensor)
                # Update x component of linear velocity
                ego_speed[0] = lin_vel[0]

        # Update position_2d from odometry data
        position_2d[0] = odom_msg[0]  # x position
        position_2d[1] = odom_msg[1]  # y position
        position_2d[2] = odom_msg[5]  # qz (orientation z component)

        return new_transform, ego_speed, position_2d


class CompassGoalChecker:
    """Checks if the robot has reached the goal."""

    def __init__(self, goal_tolerance=1.0):
        self.goal_tolerance = goal_tolerance

    @annotate.method(node_name="compass_goal_checker", export_with="torch")
    def check(self, position_2d: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """
        Check if goal is reached.

        Args:
            position_2d: Current robot position [x, y, theta]
            goal: Goal position [x, y, z, qx, qy, qz, qw]

        Returns:
            (is_reached: bool tensor, stop_cmd: zero command if reached)
        """

        # Extract 2D goal position
        goal_2d = torch.stack([goal[0], goal[1], goal[5]])  # [x, y, qz]

        # Calculate distance
        distance = torch.norm(position_2d - goal_2d)

        # Check if within tolerance
        is_reached = distance < self.goal_tolerance

        return is_reached


class CompassRouteCalculator:
    """Calculates route vectors for navigation."""

    def __init__(self, num_route_points=11, route_vector_size=4):
        self.num_route_points = num_route_points
        self.route_vector_size = route_vector_size

    def upsample_points(self, start_point: torch.Tensor, end_point: torch.Tensor,
                        max_distance: float) -> list[torch.Tensor]:
        """Upsample path between two points."""
        dist = torch.norm(end_point - start_point)

        if dist <= max_distance:
            return [start_point, end_point]

        num_segments = int(torch.ceil(dist / max_distance).item())
        upsampled_points = []

        for i in range(num_segments + 1):
            t = i / num_segments
            point = start_point * (1 - t) + end_point * t
            upsampled_points.append(point)

        return upsampled_points

    def transform_pose(self, pose: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
        """Transform pose using transform matrix."""
        # Extract transform components
        t_x, t_y, t_z = transform[0], transform[1], transform[2]
        # Only need qz and qw for 2D rotation
        t_qz, t_qw = transform[5], transform[6]

        # Extract pose components
        pose_x, pose_y, pose_z = pose[0], pose[1], pose[2]
        pose_qx, pose_qy, pose_qz, pose_qw = pose[3], pose[4], pose[5], pose[6]

        # Apply 2D rotation using quaternion yaw
        t_angle = 2 * torch.atan2(t_qz, t_qw)
        cos_t = torch.cos(t_angle)
        sin_t = torch.sin(t_angle)

        # Rotate position
        rotated_x = pose_x * cos_t - pose_y * sin_t
        rotated_y = pose_x * sin_t + pose_y * cos_t
        rotated_z = pose_z

        # Apply translation
        transformed_x = rotated_x + t_x
        transformed_y = rotated_y + t_y
        transformed_z = rotated_z + t_z

        return torch.stack([transformed_x, transformed_y, transformed_z,
                           pose_qx, pose_qy, pose_qz, pose_qw])

    @annotate.method(node_name="compass_route_calculator", export_with="torch")
    def calculate(self, goal_pose: torch.Tensor, transform: torch.Tensor,
                  max_distance: float = 1.0) -> torch.Tensor:
        """
        Calculate route vectors to goal.

        Args:
            goal_pose: Goal pose [x, y, z, qx, qy, qz, qw]
            transform: Transform from robot to goal frame [tx, ty, tz, qx, qy, qz, qw]
            max_distance: Maximum distance between route points

        Returns:
            Route vectors tensor [num_route_points-1, route_vector_size]
        """
        # Transform goal to robot frame
        transformed_pose = self.transform_pose(goal_pose, transform)

        # Extract 2D position
        goal_position = transformed_pose[0:2]
        start_point = torch.zeros(2, device=goal_position.device)

        # Upsample route
        route_poses = self.upsample_points(
            start_point, goal_position, max_distance)

        # Limit number of poses
        num_poses = min(len(route_poses), self.num_route_points)

        if num_poses == 0:
            return torch.zeros((self.num_route_points - 1, self.route_vector_size))

        # Select route points and extend with last point if needed
        indices = list(range(num_poses))
        indices.extend([num_poses - 1] *
                       (self.num_route_points - len(indices)))

        # Extract positions
        selected_positions = [route_poses[idx] for idx in indices]

        # Create route vectors
        route_vectors = torch.zeros(
            (self.num_route_points - 1, self.route_vector_size))

        for idx in range(self.num_route_points - 1):
            # Start point
            route_vectors[idx, 0:2] = selected_positions[idx]
            route_vectors[idx, 2:4] = selected_positions[idx + 1]  # End point

        return route_vectors


class CompassCommandProcessor(torch.nn.Module):
    """Processes navigation commands and applies velocity limits."""

    def __init__(self, max_linear_speed_x=0.8, max_linear_speed_y=0.5, max_angular_speed=1.0):
        super().__init__()
        self.max_linear_speed_x = max_linear_speed_x
        self.max_linear_speed_y = max_linear_speed_y
        self.max_angular_speed = max_angular_speed

    def process(self, nav_commands: torch.Tensor) -> torch.Tensor:
        """
        Apply velocity limits to navigation commands.

        Args:
            nav_commands: Raw navigation commands [6 elements]

        Returns:
            Limited velocity commands [linear_x, linear_y, angular_z]
        """
        cmd_vel = torch.zeros(3)

        # Apply velocity limits
        cmd_vel[0] = torch.clamp(nav_commands[0], -self.max_linear_speed_x,
                                 self.max_linear_speed_x)
        cmd_vel[1] = torch.clamp(nav_commands[1], -self.max_linear_speed_y,
                                 self.max_linear_speed_y)
        cmd_vel[2] = torch.clamp(
            nav_commands[5], -self.max_angular_speed, self.max_angular_speed)

        return cmd_vel

    def forward(self, nav_commands: torch.Tensor) -> torch.Tensor:
        return self.process(nav_commands)


class CompassNavigationModel:
    """Main compass navigation model that orchestrates the full pipeline."""

    def __init__(self, mobility_model_path: str, device: str = "cuda"):
        # Check if CUDA is available and fallback to CPU if not
        if device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            device = "cpu"
        self.device = device

        # Initialize all processors
        self.image_processor = CompassImageProcessor()
        self.odom_processor = CompassOdometryProcessor()
        self.goal_checker = CompassGoalChecker()
        self.route_calculator = CompassRouteCalculator()
        self.cmd_processor = CompassCommandProcessor()

        # Load mobility model
        self.mobility_model = torch.jit.load(
            mobility_model_path, map_location=device)
        print(f"Loaded mobility model from: {mobility_model_path}")

        # Initialize state variables
        self.speed = torch.zeros(1, dtype=torch.float32, device=device)
        self.position_2d = torch.zeros(3, dtype=torch.float32, device=device)
        self.transform = torch.tensor([0.05, 0.02, 0.01, 0.99, 0.0, 0.01, 0.0, 999.0, 0.0],
                                      dtype=torch.float32, device=device)
        self.action = torch.zeros((1, 6), dtype=torch.float32, device=device)
        self.history = torch.zeros(
            (1, 1024), dtype=torch.float32, device=device)
        self.sample = torch.zeros((1, 512), dtype=torch.float32, device=device)

        self.stop_cmd = torch.zeros(6, dtype=torch.float32, device=device)

    def run_navigation_pipeline(self, raw_image: torch.Tensor, odom_data: torch.Tensor,
                                goal_pose: torch.Tensor, transform: torch.Tensor,
                                route_transform: torch.Tensor) -> torch.Tensor:
        """
        Run the complete navigation pipeline.

        Args:
            raw_image: Raw camera image [height, width, 3], uint8
            odom_data: Odometry data [15 elements] 
            goal_pose: Goal pose [x, y, z, qx, qy, qz, qw]
            transform: Current transform [9 elements]
            route_transform: Transform for route calculation [7 elements]

        Returns:
            Final velocity commands [linear_x, linear_y, angular_z]
        """
        print("Running compass navigation pipeline...")

        # Step 1: Process odometry
        prev_transform = self.transform.clone()
        self.transform, self.speed, self.position_2d = self.odom_processor.process(
            odom_data, transform, prev_transform, self.speed, self.position_2d)

        # Step 2: Check if goal is reached
        is_reached = self.goal_checker.check(
            self.position_2d, goal_pose)

        # Step 3: Process image
        processed_image = self.image_processor.process(raw_image)

        # Step 4: Calculate route
        route_vectors = self.route_calculator.calculate(
            goal_pose, route_transform)

        # Step 5: Prepare inputs for mobility model
        # Add batch and time dimensions
        start_time = time.time()
        with annotate.block(node_name="process_and_run_inference", export_with="torch",
                            inputs=["processed_image",
                                    "route_vectors", "self.speed"],
                            outputs=["action_output"],
                            register_buffers=["self.action",
                                              "self.history", "self.sample"],
                            environment_constants=['self.mobility_model']):
            image_input = processed_image.unsqueeze(0).unsqueeze(
                0).to(self.device)  # [1, 1, 3, H, W]
            route_input = route_vectors.unsqueeze(0).unsqueeze(
                0).to(self.device)    # [1, 1, 29, 4]
            speed_input = self.speed.unsqueeze(0).unsqueeze(
                0).to(self.device)       # [1, 1, 1]

            # Step 6: Run mobility model
            print("Running mobility model inference...")

            action_output, history_output, sample_output = self.mobility_model(
                image_input, route_input, speed_input,
                self.action, self.history, self.sample)
            # Update state
            self.action = action_output
            self.history = history_output
            self.sample = sample_output

        end_time = time.time()
        print(
            f"Mobility model inference took {(end_time - start_time) * 1000:.2f} ms")

        # Step 7: Post-process commands
        with annotate.block(node_name="post_process_commands", export_with="torch",
                            inputs=["action_output", "is_reached"],
                            outputs=["cmd"],
                            environment_constants=['self.cmd_processor']):
            if is_reached:
                print("Goal reached! Stopping robot.")
                cmd = self.cmd_processor(self.stop_cmd)
            else:
                raw_commands = action_output.squeeze()
                cmd = self.cmd_processor(raw_commands)

        return cmd


def create_test_data(device: str = "cuda"):
    """Create test input data."""
    # Check if CUDA is available and fallback to CPU if not
    if not torch.cuda.is_available():
        device = "cpu"

    # Create test image
    torch.manual_seed(42)
    test_image = torch.randint(0, 256, (720, 1280, 3), dtype=torch.uint8)

    # Create test odometry
    test_odom = torch.tensor([
        1.0, 0.5, 0.0,         # position
        0.0, 0.0, 0.0, 1.0,    # orientation quaternion
        0.5, 0.0, 0.0,         # linear velocity
        0.0, 0.0, 0.1,         # angular velocity
        1000.0, 0.0            # timestamp
    ], dtype=torch.float32, device=device)

    # Create test transform
    test_transform = torch.tensor([
        1.0, 0.5, 0.0,         # position
        0.0, 0.0, 0.0, 1.0,    # orientation
        1000.0, 0.0            # timestamp
    ], dtype=torch.float32, device=device)

    # Create goal pose
    goal_pose = torch.tensor(
        [5.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float32, device=device)

    # Create route transform
    route_transform = torch.tensor(
        [-1.0, -0.5, -0.5, 0.0, 0.0, 0.0, 1.0], dtype=torch.float32, device=device)

    return test_image, test_odom, test_transform, goal_pose, route_transform


def main():
    """Main function."""
    abs_path = os.path.abspath(__file__)
    print(f"Absolute path of this file: {abs_path}")
    parser = argparse.ArgumentParser(
        description='Run compass navigation model in native Python')
    parser.add_argument('--mobility-model', type=str,
                        default=os.path.join(os.path.dirname(
                            abs_path), 'models', 'digit_mobility.jit'),
                        help='Path to the mobility model JIT file')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on (cuda/cpu)')
    args = parser.parse_args()

    try:
        # Initialize the compass model
        print("Initializing compass navigation model...")
        compass_model = CompassNavigationModel(
            mobility_model_path=args.mobility_model,
            device=args.device
        )

        # Create test data
        print("Creating test input data...")
        test_image, test_odom, test_transform, goal_pose, route_transform = create_test_data(
            device=compass_model.device)

        annotate.start(name="sample_compass_navigation_pipeline", verbose=True)
        # Run navigation pipeline
        print("Running navigation pipeline...")
        final_commands = compass_model.run_navigation_pipeline(
            test_image, test_odom, goal_pose, test_transform, route_transform)
        annotate.stop()
        annotate.compile_graph()

        # Print results
        print("\n=== Navigation Pipeline Complete ===")
        print("Final velocity commands: [{:.6f}, {:.6f}, {:.6f}]".format(
            final_commands[0].item(), final_commands[1].item(), final_commands[2].item()))
        print(f"Linear X: {final_commands[0].item():.6f} m/s")
        print(f"Linear Y: {final_commands[1].item():.6f} m/s")
        print(f"Angular Z: {final_commands[2].item():.6f} rad/s")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
