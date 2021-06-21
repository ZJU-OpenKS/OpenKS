from typing import Callable, Optional, Tuple

from .graph import MMGraph


class Pipeline:
    def __init__(self, *tasks: Callable[[MMGraph], MMGraph]):
        self.tasks: Tuple[Callable[[MMGraph], MMGraph], ...] = tasks

    def run(self, graph: Optional[MMGraph] = None) -> MMGraph:
        if graph is None:
            graph = MMGraph()

        for task in self.tasks:
            graph = task(graph)

        return graph
