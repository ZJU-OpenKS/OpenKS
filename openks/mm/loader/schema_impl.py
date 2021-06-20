from .schema import Entity, Relation


class ImageEntity(Entity):
    _concept = "image"

    # properties
    filename: str

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


class SemanticallySimilar(Relation):
    _concept = "semantically_similar"

    # properties
    score: float
