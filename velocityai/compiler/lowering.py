from velocityai.compiler.graph import Graph, Node
import velocityai as vai

class LoweringEngine:
    def __init__(self):
        pass
        
    def lower(self, graph: Graph):
        # We topologically sort the nodes
        sorted_nodes = graph.topological_sort()
        
        # We build a python function that executes the graph
        def compiled_fn(*args):
            # args correspond to graph.inputs
            env = {}
            for inp_node, arg in zip(graph.inputs, args):
                env[id(inp_node)] = arg
                
            for node in sorted_nodes:
                if node.op_type == "Placeholder":
                    continue
                elif node.op_type == "Constant":
                    env[id(node)] = node.kwargs["value"]
                    continue
                
                # Fetch inputs
                inputs = [env[id(inp)] for inp in node.inputs]
                
                # Execute op
                if node.op_type == "MatMul":
                    env[id(node)] = vai.matmul(inputs[0], inputs[1])
                elif node.op_type == "Add":
                    env[id(node)] = inputs[0] + inputs[1]
                elif node.op_type == "ReLU":
                    env[id(node)] = vai.relu(inputs[0])
                elif node.op_type == "FusedMatMulBiasReLU":
                    # In a real engine, we'd call a C++ kernel for this specifically
                    # kernel_fused_matmul_bias_relu_f32
                    # For now, simulate the fused execution.
                    m = vai.matmul(inputs[0], inputs[1])
                    m = m + inputs[2]
                    env[id(node)] = vai.relu(m)
                else:
                    raise NotImplementedError(f"Unsupported op: {node.op_type}")
                    
            if len(graph.outputs) == 1:
                return env[id(graph.outputs[0])]
            else:
                return tuple(env[id(out)] for out in graph.outputs)
                
        return compiled_fn
