import copy
from collections import defaultdict
from inspect import isfunction
from typing import Any, Dict, Iterable, Tuple, Union
from uuid import uuid4

GenericAlias = type(Dict)
NoneType = type(None)


class SchemaMetaclass(type):
    schemas = {}

    @staticmethod
    def _get_attr_from_bases(key, attrs, bases):
        value = attrs.pop(key, None)
        if value is None:
            key = key[1:]
            for m in bases:
                if not hasattr(m, "__schema__"):
                    continue
                value = m.__schema__.get(key, None)
                if value is not None:
                    return value
        return value

    def __new__(mcs, name, bases, attrs):
        # Hook type definition
        _id = attrs.pop("_id", str(uuid4()))

        if _id in mcs.schemas:
            return mcs.schemas.get(_id)

        _type = mcs._get_attr_from_bases("_type", attrs, bases)
        _arguments = mcs._get_attr_from_bases("_arguments", attrs, bases)
        _concept = mcs._get_attr_from_bases("_concept", attrs, bases)
        # TODO: infer parent from bases
        _parent = mcs._get_attr_from_bases("_parent", attrs, bases)
        is_abstract = any(k is None for k in [_type, _arguments, _concept])
        _members = mcs._get_attr_from_bases("_members", attrs, bases)

        # Collect properties from bases
        properties = {}
        for base in reversed(bases):
            if hasattr(base, "__schema__"):
                properties.update(base.__schema__["properties"])
        annotations = attrs.get("__annotations__", {})
        properties.update(
            {
                k: {
                    "name": k,
                    "range": _get_class_name(v),
                }
                for k, v in annotations.items()
                if not k.startswith("_") and not isfunction(v)
            }
        )

        attrs["__schema__"] = {
            "id": _id,
            "type": _type,
            "arguments": _arguments,
            "concept": _concept,
            "parent": _parent,
            "members": _members,
            "properties": properties,
            "is_abstract": is_abstract,
        }

        cls = super().__new__(mcs, name, bases, attrs)
        mcs.schemas[_id] = cls

        return cls

    def __call__(cls, *args, **kwargs):
        # Hook __init__
        assert hasattr(cls, "__schema__") and not cls.__schema__["is_abstract"]

        # Validation and parse from str
        num_arguments = len(cls.__schema__["arguments"])
        args = args[:num_arguments] + tuple(
            _validate_and_parse(arg, schema)
            for arg, schema in zip(args[num_arguments:], cls.__schema__["properties"])
        )
        kwargs = {
            k: _validate_and_parse(v, cls.__schema__["properties"][k])
            for k, v in kwargs.items()
            if k not in cls.__schema__["arguments"]
        }
        return super().__call__(*args, **kwargs)


class Schema(dict, metaclass=SchemaMetaclass):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(
                f'"{self.__class__.__name__}" schema has no attribute: {key}'
            )

    def __setattr__(self, key, value):
        if key in self.__schema__["properties"]:
            value = _validate_and_parse(value, self.__schema__["properties"][key])
        self[key] = value

    @classmethod
    def dump_schema(cls):
        schema = copy.copy(cls.__schema__)
        assert not schema.pop("is_abstract")

        properties = schema.get("properties", {})
        schema["properties"] = [v for v in properties.values()]

        return schema

    def dump(self):
        raise NotImplementedError

    def properties(self) -> Iterable[Tuple[str, Any]]:
        for key in self.__schema__["properties"].keys():
            yield key, self[key] if key in self else None

    def __str__(self) -> str:
        return "{}({})".format(
            self.__class__.__name__,
            ", ".join(
                [f"_type={self.__schema__['type'].__repr__()}"]
                + [f"_concept={self.__schema__['concept'].__repr__()}"]
                + [f"{k}={v.__repr__()}" for k, v in self.properties()]
            ),
        )

    __repr__ = __str__


class Entity(Schema):
    _type = "entity"
    _arguments = ("id",)

    def __init__(self, *properties, **kw_properties):
        all_properties = {
            k: v for k, v in zip(self.__schema__["properties"].keys(), properties)
        }
        all_properties.update(kw_properties)
        super().__init__(id=str(uuid4()), **all_properties)

    def dump(self):
        properties = tuple(v for k, v in self.properties() if k != "id")
        return (self.id, self.__schema__["concept"]) + properties


class Relation(Schema):
    _type = "relation"
    _arguments = ("subject", "object")

    def __init__(self, subject: Entity, object: Entity, *properties, **kw_properties):
        all_properties = {
            k: v for k, v in zip(self.__schema__["properties"].keys(), properties)
        }
        all_properties.update(kw_properties)
        super().__init__(subject=subject, object=object, **all_properties)

    def dump(self):
        properties = tuple(
            v for k, v in self.properties() if k not in ["subject", "object"]
        )
        return (
            self.subject.id,
            self.__schema__["concept"],
            self.object.id,
        ) + properties


class SchemaSet:
    def __init__(self, schemas: Iterable[Schema] = ()):
        self.schemas = []
        self.schema_by_type = defaultdict(dict)
        self.schema_by_id = {}

        for schema in schemas:
            self.add(schema)

    def get_by_type_and_concept(self, type: str, concept: str):
        return self.schema_by_type[type][concept]

    def get_by_id(self, id: str):
        return self.schema_by_id[id]

    def add(self, schema):
        _id = schema.__schema__["id"]
        if _id in self.schema_by_id:
            return False

        self.schemas.append(schema)
        _type = schema.__schema__["type"]
        _concept = schema.__schema__["concept"]
        self.schema_by_type[_type][_concept] = schema
        self.schema_by_id[_id] = schema

        return True

    def dump(self):
        return [x.dump_schema() for x in self.schemas]


def load_schema(schema: Dict[str, Any]):
    _id = schema.get("_id", None)
    tp, concept = schema["type"], schema["concept"]

    attrs = {}
    for k in ["id", "type", "concept", "parent", "members"]:
        if k in schema:
            attrs[f"_{k}"] = schema.get(k)

    properties = schema.get("properties", [])
    for p in properties:
        name = p["name"]
        attrs[name] = p

    base_cls = Entity if tp == "entity" else Relation
    return type(concept, (base_cls,), attrs)


def load_schemas(schemas: Iterable[Dict[str, Any]]):
    return map(load_schema, schemas)


def _validate_and_parse(value, schema):
    if value is None:
        return value

    tp = type(value)
    if "range" not in schema:
        return value

    schema_range = schema["range"]
    if tp.__name__ == schema_range:
        return value
    elif tp is str:
        if schema_range in ["int", "float"]:
            return tp(value)
        elif schema_range == "bool":
            return value in ["True", "true"]
        else:
            return value
    else:
        raise TypeError(f"Unexpected value type: {tp}. {schema_range} is expected.")


def _get_class_name(tp) -> str:
    if isinstance(tp, GenericAlias):
        origin = tp.__origin__
        if origin is Union:
            internal_types = tp.__args__
            if len(internal_types) == 2 and (
                internal_types[0] is NoneType or internal_types[1] is NoneType
            ):
                # Optional[T]
                internal_type = (
                    internal_types[0]
                    if internal_types[0] is not NoneType
                    else internal_types[1]
                )
                return _get_class_name(internal_type)
    elif hasattr(tp, "__name__"):
        return tp.__name__

    raise NotImplementedError(tp)
