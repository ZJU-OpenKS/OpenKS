from abc import abstractmethod, ABCMeta
from typing import Union
from pathlib import Path

from .graph import MMGraph, ImageEntity


class DataLoader(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, graph: MMGraph) -> MMGraph:
        ...


class GlobDataLoader(DataLoader):
    def __init__(self, root_dir: Union[str, Path], pattern: str = "*"):
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)

        self.root_dir: Path = root_dir
        self.pattern = pattern

    def __call__(self, graph: MMGraph) -> MMGraph:
        for file_name in self.root_dir.glob(self.pattern):
            graph.add_entity(ImageEntity(file_name=file_name))

        return graph
