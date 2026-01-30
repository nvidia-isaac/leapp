from leapp import annotate
import numpy as np
import torch
import unittest
import os
import shutil


class TestNumpyCompatibilityTracedTensor(unittest.TestCase):
    """Test the numpy compatibility of the TracedTensor."""

    def setUp(self):
        self.TEST_GRAPH_NAME = "test_graph"

    def tearDown(self):
        if os.path.exists(self.TEST_GRAPH_NAME):
            shutil.rmtree(self.TEST_GRAPH_NAME)

    def test_basic_numpy_function(self):
        tensor = torch.tensor([1.0, 2.0, 3.0])

        annotate.start(name=self.TEST_GRAPH_NAME)
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


class TestNumpyFunctionsWithPatchesApplied(unittest.TestCase):
    """Test that regular numpy functions work correctly when patches are applied.
    
    This verifies that torch.from_numpy, torch.as_tensor, and torch.tensor
    work correctly with regular numpy arrays between start() and stop(),
    even though the global patches are applied.
    """

    def setUp(self):
        self.TEST_GRAPH_NAME = "test_graph"

    def tearDown(self):
        if os.path.exists(self.TEST_GRAPH_NAME):
            shutil.rmtree(self.TEST_GRAPH_NAME)

    def test_numpy_to_torch_between_start_stop(self):
        """Test torch.from_numpy works with regular numpy arrays during annotation."""
        annotate.start(name=self.TEST_GRAPH_NAME)
        
        # Regular numpy arrays (not TracedTensors)
        np_array1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        np_array2 = np.array([[1, 2], [3, 4]], dtype=np.float64)
        
        # These should work normally via the patched functions
        tensor1 = torch.from_numpy(np_array1)
        tensor2 = torch.from_numpy(np_array2)
        
        # Verify the conversions worked correctly
        self.assertIsInstance(tensor1, torch.Tensor)
        self.assertIsInstance(tensor2, torch.Tensor)
        self.assertEqual(tensor1.shape, torch.Size([3]))
        self.assertEqual(tensor2.shape, torch.Size([2, 2]))
        self.assertTrue(torch.allclose(tensor1, torch.tensor([1.0, 2.0, 3.0])))
        
        annotate.stop()

    def test_as_tensor_between_start_stop(self):
        """Test torch.as_tensor works with regular numpy arrays during annotation."""
        annotate.start(name=self.TEST_GRAPH_NAME)
        
        np_array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        
        # torch.as_tensor should work normally
        tensor = torch.as_tensor(np_array)
        
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, torch.Size([3]))
        self.assertTrue(torch.allclose(tensor, torch.tensor([1.0, 2.0, 3.0])))
        
        # Test with dtype conversion
        tensor_int = torch.as_tensor(np_array, dtype=torch.int32)
        self.assertEqual(tensor_int.dtype, torch.int32)
        
        annotate.stop()

    def test_torch_tensor_between_start_stop(self):
        """Test torch.tensor works with regular numpy arrays during annotation."""
        annotate.start(name=self.TEST_GRAPH_NAME)
        
        np_array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        
        # torch.tensor should work normally
        tensor = torch.tensor(np_array)
        
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, torch.Size([3]))
        self.assertTrue(torch.allclose(tensor, torch.tensor([1.0, 2.0, 3.0])))
        
        # Test with dtype conversion
        tensor_int = torch.tensor(np_array, dtype=torch.int64)
        self.assertEqual(tensor_int.dtype, torch.int64)
        
        annotate.stop()

    def test_numpy_operations_between_start_stop(self):
        """Test that standard numpy operations work correctly during annotation."""
        annotate.start(name=self.TEST_GRAPH_NAME)
        
        # Standard numpy operations should work unaffected
        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([4.0, 5.0, 6.0])
        
        result_add = np.add(arr1, arr2)
        result_mul = np.multiply(arr1, arr2)
        result_sin = np.sin(arr1)
        result_concat = np.concatenate([arr1, arr2])
        
        # Verify numpy operations produced correct results
        np.testing.assert_array_almost_equal(result_add, [5.0, 7.0, 9.0])
        np.testing.assert_array_almost_equal(result_mul, [4.0, 10.0, 18.0])
        np.testing.assert_array_almost_equal(result_sin, np.sin([1.0, 2.0, 3.0]))
        np.testing.assert_array_almost_equal(result_concat, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        
        # Convert final result to torch
        tensor = torch.from_numpy(result_concat)
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, torch.Size([6]))
        
        annotate.stop()
