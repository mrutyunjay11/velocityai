from velocityai.compiler.passes.base_pass import BasePass
from velocityai.compiler.graph import Graph

class QuantizationPass(BasePass):
    def run(self, graph: Graph) -> Graph:
        # To be implemented: post-training INT8/INT4 quantization
        return graph
