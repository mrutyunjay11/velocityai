import contextlib
from velocityai.compiler.graph import Graph, Node

_ACTIVE_TRACER = None

class Tracer:
    def __init__(self):
        self.graph = Graph()
        self.tensor_to_node = {}

    def add_input(self, tensor, name="input"):
        node = self.graph.add_input(name, tensor.shape, tensor.dtype)
        self.tensor_to_node[id(tensor)] = node
        return node
        
    def record_op(self, op_type, inputs, output_tensor, **kwargs):
        node_inputs = []
        for inp in inputs:
            if id(inp) in self.tensor_to_node:
                node_inputs.append(self.tensor_to_node[id(inp)])
            else:
                # Constant
                const_node = Node("Constant", [], {"value": inp})
                self.graph.add_node(const_node)
                node_inputs.append(const_node)
                
        node = Node(op_type, node_inputs, kwargs)
        node.shape = output_tensor.shape
        node.dtype = output_tensor.dtype
        self.graph.add_node(node)
        self.tensor_to_node[id(output_tensor)] = node
        return node
        
    def set_outputs(self, tensors):
        out_nodes = [self.tensor_to_node[id(t)] for t in tensors]
        self.graph.set_outputs(out_nodes)

@contextlib.contextmanager
def trace():
    global _ACTIVE_TRACER
    old_tracer = _ACTIVE_TRACER
    _ACTIVE_TRACER = Tracer()
    try:
        yield _ACTIVE_TRACER
    finally:
        _ACTIVE_TRACER = old_tracer

def get_active_tracer():
    return _ACTIVE_TRACER
