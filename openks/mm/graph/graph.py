import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

from .schema import Entity, Relation, SchemaSet, load_schemas
from .utils import remove_null


class MMGraph:
    def __init__(
        self,
        schemas: Optional[SchemaSet] = None,
        entities: Optional[List[Entity]] = None,
        relations: Optional[List[Relation]] = None,
    ):
        if schemas is None:
            schemas = SchemaSet()
        if entities is None:
            entities = []
        if relations is None:
            relations = []
        self.schemas: SchemaSet = schemas
        self.entities: Dict[str, Entity] = {entity.id: entities for entity in entities}
        self.relations: List[Relation] = relations

    @classmethod
    def load(cls, path: Union[str, Path]):
        if isinstance(path, str):
            path = Path(path)

        with open(path / "schema.json") as f:
            schemas = json.load(f)
        schemas = load_schemas(schemas)
        schemas = SchemaSet(schemas)

        with open(path / "entities") as f:
            entities = f.read().split("\n")
        entities = map(lambda x: x.split("\t"), entities)
        entities = filter(lambda x: len(x) >= 2, entities)
        entities = [
            schemas.get_by_type_and_concept("entity", x[1])(x[0], *x[2:])
            for x in entities
        ]
        entity_by_id = {x.id: x for x in entities}

        with open(path / "triples") as f:
            triples = f.read().split("\n")
        triples = map(lambda x: x.split("\t"), triples)
        triples = filter(lambda x: len(x) >= 3, triples)
        relations = [
            schemas.get_by_type_and_concept("relation", x[1])(
                entity_by_id[x[0]], entity_by_id[x[2]], *x[3:]
            )
            for x in triples
        ]

        return cls(schemas, entities, relations)

    def save(self, path: Union[str, Path]):
        if isinstance(path, str):
            path = Path(path)

        if not path.exists():
            path.mkdir()

        with open(path / "schema.json", "w") as f:
            schemas = self.schemas.dump()
            schemas = remove_null(schemas)
            json.dump(schemas, f)

        with open(path / "entities", "w") as f:
            for entity in self.entities.values():
                entity = map(str, entity.dump())
                f.write("\t".join(entity) + "\n")

        with open(path / "triples", "w") as f:
            for relation in self.relations:
                relation = map(str, relation.dump())
                f.write("\t".join(relation) + "\n")

    def add_entity(self, entity: Entity):
        self.schemas.add(entity.__class__)
        if entity.id not in self.entities:
            self.entities[entity.id] = entity

    def add_entities(self, entities: Iterable[Entity]):
        for entity in entities:
            self.add_entity(entity)

    def add_relation(self, relation: Relation):
        self.schemas.add(relation.__class__)
        self.add_entities([relation.subject, relation.object])
        self.relations.append(relation)

    def add_relations(self, relations: Iterable[Relation]):
        for relation in relations:
            self.add_relation(relation)

    def get_entity_by_id(self, id: str):
        return self.entities[id]

    def get_entities_by_concept(self, concept: str):
        return filter(
            lambda x: x.__schema__["concept"] == concept, self.entities.values()
        )


def main():
    # g = MMGraph.load("openks/data/company-kg")
    # g.save("company-kg")
    g = MMGraph()
    from .schema_impl import ImageEntity, ImageViewEntity, SemanticallySimilar

    e1 = ImageEntity(filename="1.png")
    e2 = ImageEntity(filename="2.png")
    g.add_entities([e1, e2])
    e3 = ImageViewEntity(image=e1, x0=0, y0=10, x1=100, y1=200)
    g.add_entity(e3)

    r1 = SemanticallySimilar(e1, e2, score=10)
    g.add_relation(r1)

    r2 = SemanticallySimilar(e1, e3, score=100)
    g.add_relation(r2)

    g.save("../test-kg")

    print(list(g.get_entities_by_concept("image")))
    print(list(g.get_entities_by_concept("image_view")))


if __name__ == "__main__":
    main()
