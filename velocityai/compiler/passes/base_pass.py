from abc import ABC, abstractmethod
from velocityai.compiler.graph import Graph

class BasePass(ABC):
    @abstractmethod
    def run(self, graph: Graph) -> Graph:
        pass
