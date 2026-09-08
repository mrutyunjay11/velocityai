from velocityai.compiler.passes.base_pass import BasePass
from velocityai.compiler.graph import Graph

class DeadCodeEliminationPass(BasePass):
    def run(self, graph: Graph) -> Graph:
        # Reachability analysis from outputs
        visited = set()
        
        def visit(n):
            if n in visited:
                return
            visited.add(n)
            for inp in n.inputs:
                visit(inp)
                
        for out in graph.outputs:
            visit(out)
            
        graph.nodes = [n for n in graph.nodes if n in visited]
        return graph
