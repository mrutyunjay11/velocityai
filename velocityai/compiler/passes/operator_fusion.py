from velocityai.compiler.passes.base_pass import BasePass
from velocityai.compiler.graph import Graph, Node

class OperatorFusionPass(BasePass):
    def run(self, graph: Graph) -> Graph:
        # Simple pattern matching for fusion
        # MatMul -> Add (Bias) -> ReLU  => FusedMatMulBiasReLU
        nodes = graph.topological_sort()
        
        for node in nodes:
            # Match Add after MatMul
            if node.op_type == "Add" and len(node.inputs) == 2:
                matmul_node = None
                bias_node = None
                if node.inputs[0].op_type == "MatMul":
                    matmul_node = node.inputs[0]
                    bias_node = node.inputs[1]
                elif node.inputs[1].op_type == "MatMul":
                    matmul_node = node.inputs[1]
                    bias_node = node.inputs[0]
                    
                if matmul_node:
                    # Found MatMul + Bias. Check if outputs to ReLU.
                    relu_child = None
                    if len(node.outputs) == 1 and node.outputs[0].op_type == "ReLU":
                        relu_child = node.outputs[0]
                        
                    if relu_child:
                        # Fused MatMul + Bias + ReLU
                        fused_node = Node("FusedMatMulBiasReLU", 
                                        matmul_node.inputs + [bias_node], 
                                        kwargs=matmul_node.kwargs)
                        fused_node.shape = relu_child.shape
                        fused_node.dtype = relu_child.dtype
                        
                        # Replace relu_child's consumers to read from fused_node
                        for consumer in relu_child.outputs:
                            for i, inp in enumerate(consumer.inputs):
                                if inp == relu_child:
                                    consumer.inputs[i] = fused_node
                                    fused_node.outputs.append(consumer)
                        
                        # If relu_child was a graph output
                        for i, out in enumerate(graph.outputs):
                            if out == relu_child:
                                graph.outputs[i] = fused_node
                                
                        graph.nodes.append(fused_node)
                        # We don't strictly remove the old nodes from the list yet,
                        # DeadCodeElimination pass will clean them up.
                        
        return graph
