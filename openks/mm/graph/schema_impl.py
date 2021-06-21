from PIL import Image

from .schema import Entity, Relation
from .utils import read_image


class ImageEntity(Entity):
    _concept = "image"

    # properties
    file_name: str

    @property
    def data(self):
        if not hasattr(self, "_data"):
            # TODO: support relative path
            self._data = read_image(self.file_name)

        return self._data

    def view(self, x0: int, y0: int, x1: int, y1: int):
        return ImageViewEntity(image=self, x0=x0, y0=y0, x1=x1, y1=y1)


class ImageViewEntity(Entity):
    _concept: str = "image_view"
    _parent = "image"

    # properties
    image_id: str
    x0: int
    y0: int
    x1: int
    y1: int

    def __init__(self, image: ImageEntity, x0: int, y0: int, x1: int, y1: int):
        super().__init__(image_id=image.id, x0=x0, y0=y0, x1=x1, y1=y1)

        self._src_image = image

    @property
    def data(self):
        img = self._src_image.data
        if len(img.shape) <= 3:
            return img[self.y0: self.y1, self.x0: self.x1]
        else:
            return img[..., self.y0: self.y1, self.x0: self.x1, :]


class SemanticallySimilar(Relation):
    _concept = "semantically_similar"

    # properties
    score: float
