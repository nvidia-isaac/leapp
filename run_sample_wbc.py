from leapp.inference_manager import InferenceManager
from examples.wbc_obj import WBC

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


def main():
    wbc = WBC()
    wbc_policy = InferenceManager(
        "sample_wbc_graph/sample_wbc_graph.yaml", verbose=True)
    wbc_policy.set_input_value('concatenate_and_run_model', 'previous_actions', torch.randn(
        19, device=DEVICE, dtype=DTYPE))

    for i in range(10):
        inputs = {
            'concatenate_and_run_model/velocity_commands': torch.randn(3, device=DEVICE, dtype=DTYPE),
            'concatenate_and_run_model/joint_vel': torch.randn(19, device=DEVICE, dtype=DTYPE),
            'process_joint_pos/joint_pos': torch.randn(19, device=DEVICE, dtype=DTYPE),
            'process_odom/lin_vel_I': torch.randn(3, device=DEVICE, dtype=DTYPE),
            'process_odom/ang_vel_I': torch.randn(3, device=DEVICE, dtype=DTYPE),
            'process_odom/q_IB': torch.randn(4, device=DEVICE, dtype=DTYPE),
        }
        # Extract just the input names (part after '/') for the reference WBC
        inputs_for_wbc = {
            key.split('/')[1]: value for key, value in inputs.items()}
        reference_outputs = wbc.run_model(**inputs_for_wbc)

        outputs = wbc_policy.run_policy(inputs)

        # Check if outputs match reference
        output_tensor = outputs['post_process_actions/actions']
        if torch.allclose(reference_outputs, output_tensor, rtol=1e-5, atol=1e-5):
            print(f"Iteration {i}: PASS - outputs match reference")
        else:
            print(f"Iteration {i}: FAIL - outputs do not match")
            print(
                f"  Max difference: {(reference_outputs - output_tensor).abs().max().item()}")
            print(f"  Reference: {reference_outputs}")
            print(f"  Output:    {output_tensor}")


if __name__ == "__main__":
    main()
