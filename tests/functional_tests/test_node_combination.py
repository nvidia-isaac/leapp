import unittest
import torch
from leapp import annotate
from .base import LEAPPFunctionalTestBase
from leapp import MergeCfgEnum


class TestNodeMerging(LEAPPFunctionalTestBase):

    def test_combine_two_nodes_automatically(self):
        @annotate.method(export_with="torch")
        def funcA(inputA: torch.Tensor, inputA2: torch.Tensor):
            return inputA + inputA2

        @annotate.method(export_with="torch")
        def funcB(inputB: torch.Tensor):
            output1 = inputB * 2.0
            output2 = inputB * 3.0
            return output1, output2

        @annotate.method(export_with="torch")
        def funcC(inputC: torch.Tensor):
            a = inputC - 1.0
            b = inputC
            return a, b

        @annotate.method(export_with="torch")
        def funcD(inputD: torch.Tensor):
            a = inputD - 1.0
            return a

        @annotate.method(export_with="torch")
        def funcE(inputE: torch.Tensor):
            a = inputE + 1.0
            return a

        annotate.start(name="test_graph")

        out = funcA(torch.tensor([1.0, 1.0, 1.0]),
                    torch.tensor([2.0, 2.0, 2.0]))
        outb1, outb2 = funcB(out)
        outc1, outc2 = funcC(outb1)
        outd = funcD(outb2)
        outd = funcE(outd)
        annotate.stop()
        annotate.compile_graph(merge_nodes=MergeCfgEnum.AUTOMATIC)

        self.verify_num_connections(annotate, nodes=3, inputs=2, outputs=3,
                                    internal_connections=2)

    def test_combine_four_nodes_automatically(self):
        @annotate.method(export_with="torch")
        def funcA(inputA: torch.Tensor):
            return inputA + 1.0

        @annotate.method(export_with="torch")
        def funcB(inputB: torch.Tensor):
            return inputB*2.0

        @annotate.method(export_with="torch")
        def funcC(inputC: torch.Tensor):
            return inputC*3.0

        @annotate.method(export_with="torch")
        def funcD(inputD: torch.Tensor):
            return inputD*4.0

        annotate.start(name=self.TEST_GRAPH_NAME)
        out = funcA(torch.tensor([1.0, 1.0, 1.0]))
        outb1 = funcB(out)
        outc1 = funcC(outb1)
        outd = funcD(outc1)
        annotate.stop()
        annotate.compile_graph(merge_nodes=MergeCfgEnum.AUTOMATIC)

        self.verify_num_connections(annotate, nodes=1, inputs=1, outputs=1,
                                    internal_connections=0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
