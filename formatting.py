import torch
from leapp.utils import TensorDescription, ParameterFormat, describe_io


def get_packing_expr(input_formats):
    """
    Generate packing assignment expressions that convert flat tensor inputs into 
    the expected nested structure, using the original parameter names.
    
    This function handles:
    - Name transformations (e.g., "self.var1" -> "self_var1" becomes "self.var1 = self_var1")
    - List packing (e.g., "inputA = [inputA_0, inputA_1]")
    - Dict packing (e.g., 'state = {"pose": state_pose, "velocity": state_velocity}')
    - Nested structures
    
    Args:
        input_formats: List of ParameterFormat objects describing the target nested structure
    
    Returns:
        A string with newline-separated assignment statements. Empty string if no 
        assignments are needed (trivial case where name == expression).
    
    Example:
        Single parameter with list:
            inputA = [inputA_0, inputA_1]
        
        Name transformation:
            self.var1 = self_var1
        
        Multiple parameters:
            input1 = [input1_0, input1_1]
            input2 = {"a": input2_a, "b": input2_b}
    """
    
    def _generate_expr(format_item) -> str:
        """Generate expression string using tensor names (name_str for function params)."""
        if isinstance(format_item, ParameterFormat):
            return _generate_expr(format_item.formatting)
        elif isinstance(format_item, TensorDescription):
            return format_item.name
        elif isinstance(format_item, list):
            elements = [_generate_expr(item) for item in format_item]
            return "[" + ", ".join(elements) + "]"
        elif isinstance(format_item, dict):
            items = [f'"{k}": {_generate_expr(v)}' for k, v in format_item.items()]
            return "{" + ", ".join(items) + "}"
        else:
            return "None"
    
    assignments = []
    
    for param_format in input_formats:
        # Use name_raw for LHS (original name like "self.var1")
        # The RHS expression uses tensor names which match function parameters
        original_name = param_format.name  # e.g., "self.var1" 
        reconstruction = _generate_expr(param_format)
        
        # Skip trivial assignments where LHS == RHS (e.g., "x = x")
        if original_name != reconstruction:
            assignments.append(f"{original_name} = {reconstruction}")
    
    return "\n".join(assignments)


if __name__ == "__main__":
    # Test examples using describe_io
    
    print("=" * 60)
    print("TEST 1: Simple list of tensors")
    print("=" * 60)
    t1 = torch.tensor([1.0, 2.0, 3.0])
    t2 = torch.tensor([4.0, 5.0, 6.0])
    data = [t1, t2]
    io_desc, param_fmt = describe_io("inputA", "inputA", data)
    print("Data structure: list of 2 tensors")
    print(f"io_descriptions: {[d.name for d in io_desc]}")
    result = get_packing_expr([param_fmt])
    print(f"Packing:\n{result}" if result else "Packing: (none needed)")
    print()

    print("=" * 60)
    print("TEST 2: Single tensor (trivial - no packing needed)")
    print("=" * 60)
    data = torch.tensor([1.0, 2.0, 3.0])
    io_desc, param_fmt = describe_io("single", "single", data)
    print("Data structure: single tensor")
    print(f"io_descriptions: {[d.name for d in io_desc]}")
    result = get_packing_expr([param_fmt])
    print(f"Packing:\n{result}" if result else "Packing: (none needed)")
    print()

    print("=" * 60)
    print("TEST 3: Name transformation (self.var1 -> self_var1)")
    print("=" * 60)
    data = torch.tensor([1.0, 2.0, 3.0])
    # Simulate what happens when "self.var1" is transformed to "self_var1" for function params
    # describe_io uses name for tensor name and raw_name for original parameter name
    io_desc, param_fmt = describe_io("self_var1", "self.var1", data)
    print("Data structure: single tensor with name transformation")
    print(f"io_descriptions: {[d.name for d in io_desc]}")
    print(f"param_fmt.name (raw): {param_fmt.name}")
    result = get_packing_expr([param_fmt])
    print(f"Packing:\n{result}" if result else "Packing: (none needed)")
    print()

    print("=" * 60)
    print("TEST 4: Name transformation with list (self.inputs -> self_inputs)")
    print("=" * 60)
    data = [torch.tensor([1.0]), torch.tensor([2.0])]
    io_desc, param_fmt = describe_io("self_inputs", "self.inputs", data)
    print("Data structure: list with name transformation")
    print(f"io_descriptions: {[d.name for d in io_desc]}")
    print(f"param_fmt.name (raw): {param_fmt.name}")
    result = get_packing_expr([param_fmt])
    print(f"Packing:\n{result}" if result else "Packing: (none needed)")
    print()

    print("=" * 60)
    print("TEST 5: Dict of tensors")
    print("=" * 60)
    data = {"pose": torch.tensor([1.0, 2.0]), "velocity": torch.tensor([3.0, 4.0])}
    io_desc, param_fmt = describe_io("state", "state", data)
    print("Data structure: dict with keys 'pose', 'velocity'")
    print(f"io_descriptions: {[d.name for d in io_desc]}")
    result = get_packing_expr([param_fmt])
    print(f"Packing:\n{result}" if result else "Packing: (none needed)")
    print()

    print("=" * 60)
    print("TEST 6: Nested list of lists")
    print("=" * 60)
    data = [[torch.tensor([1.0]), torch.tensor([2.0])], 
            [torch.tensor([3.0]), torch.tensor([4.0])]]
    io_desc, param_fmt = describe_io("nested", "nested", data)
    print("Data structure: list of lists (2x2 tensors)")
    print(f"io_descriptions: {[d.name for d in io_desc]}")
    result = get_packing_expr([param_fmt])
    print(f"Packing:\n{result}" if result else "Packing: (none needed)")
    print()

    print("=" * 60)
    print("TEST 7: Dict containing list")
    print("=" * 60)
    data = {"positions": [torch.tensor([1.0]), torch.tensor([2.0])],
            "velocities": [torch.tensor([3.0]), torch.tensor([4.0])]}
    io_desc, param_fmt = describe_io("physics", "physics", data)
    print("Data structure: dict with list values")
    print(f"io_descriptions: {[d.name for d in io_desc]}")
    result = get_packing_expr([param_fmt])
    print(f"Packing:\n{result}" if result else "Packing: (none needed)")
    print()

    print("=" * 60)
    print("TEST 8: List of dicts")
    print("=" * 60)
    data = [{"x": torch.tensor([1.0]), "y": torch.tensor([2.0])},
            {"x": torch.tensor([3.0]), "y": torch.tensor([4.0])}]
    io_desc, param_fmt = describe_io("points", "points", data)
    print("Data structure: list of dicts")
    print(f"io_descriptions: {[d.name for d in io_desc]}")
    result = get_packing_expr([param_fmt])
    print(f"Packing:\n{result}" if result else "Packing: (none needed)")
    print()

    print("=" * 60)
    print("TEST 9: Multiple parameters with mixed transformations")
    print("=" * 60)
    data1 = [torch.tensor([1.0]), torch.tensor([2.0])]
    data2 = {"a": torch.tensor([3.0]), "b": torch.tensor([4.0])}
    io_desc1, param_fmt1 = describe_io("self_data", "self.data", data1)
    io_desc2, param_fmt2 = describe_io("config", "config", data2)
    combined_io = io_desc1 + io_desc2
    print("Data structure: two params - list (with name transform) and dict")
    print(f"io_descriptions: {[d.name for d in combined_io]}")
    result = get_packing_expr([param_fmt1, param_fmt2])
    print(f"Packing:\n{result}" if result else "Packing: (none needed)")
    print()

    print("=" * 60)
    print("TEST 10: Deeply nested structure")
    print("=" * 60)
    data = {"level1": {"level2": [torch.tensor([1.0]), torch.tensor([2.0])]}}
    io_desc, param_fmt = describe_io("deep", "deep", data)
    print("Data structure: dict -> dict -> list")
    print(f"io_descriptions: {[d.name for d in io_desc]}")
    result = get_packing_expr([param_fmt])
    print(f"Packing:\n{result}" if result else "Packing: (none needed)")
    print()

