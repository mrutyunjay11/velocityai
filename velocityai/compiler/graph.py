from typing import List, Dict, Any, Optional

class Node:
    def __init__(self, op_type: str, inputs: List['Node'], kwargs: Dict[str, Any] = None):
        self.op_type = op_type
        self.inputs = inputs
        self.kwargs = kwargs or {}
        self.outputs: List['Node'] = []
        self.shape: Optional[tuple] = None
        self.dtype = None
        
        for inp in self.inputs:
            inp.outputs.append(self)
            
    def __repr__(self):
        in_str = ", ".join([f"Node_{id(n)}" for n in self.inputs])
        return f"Node_{id(self)}(op={self.op_type}, inputs=[{in_str}])"

class Graph:
    def __init__(self):
        self.nodes: List[Node] = []
        self.inputs: List[Node] = []
        self.outputs: List[Node] = []
        
    def add_node(self, node: Node):
        self.nodes.append(node)
        return node
        
    def add_input(self, name: str, shape: tuple, dtype) -> Node:
        node = Node("Placeholder", [])
        node.kwargs['name'] = name
        node.shape = shape
        node.dtype = dtype
        self.inputs.append(node)
        self.nodes.append(node)
        return node
        
    def set_outputs(self, nodes: List[Node]):
        self.outputs = nodes

    def topological_sort(self) -> List[Node]:
        visited = set()
        sorted_nodes = []
        
        def visit(n: Node):
            if n in visited:
                return
            for inp in n.inputs:
                visit(inp)
            visited.add(n)
            sorted_nodes.append(n)
            
        for out in self.outputs:
            visit(out)
            
        return sorted_nodes
