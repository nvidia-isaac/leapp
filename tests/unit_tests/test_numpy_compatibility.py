from leapp import annotate
import numpy as np
import torch
import unittest



class TestNumpyCompatibilityTracedTensor(unittest.TestCase):
    """Test the numpy compatibility of the TracedTensor."""

    def test_basic_numpy_function(self):
        tensor = torch.tensor([1.0, 2.0, 3.0])

        annotate.start('test')
        tensor = annotate.input_tensors({'tensor': tensor}, 'basic_numpy_function')

        # numpy operations
        numpy_val = tensor.numpy()
        numpy_val1 = numpy_val + 1
        numpy_val2 = numpy_val - 1
        numpy_val3 = numpy_val1 * numpy_val2
        numpy_val4 = numpy_val3 / 2
        torch_tensors = [torch.from_numpy(numpy_val) for numpy_val in [numpy_val1, numpy_val2, numpy_val3, numpy_val4]]

        annotate.output_tensors('basic_numpy_function', {'tensor': torch_tensors}, export_with="jit")
        annotate.stop()
        annotate.compile_graph(visualize=False)
        print(torch_tensors)
        import pdb; pdb.set_trace()
