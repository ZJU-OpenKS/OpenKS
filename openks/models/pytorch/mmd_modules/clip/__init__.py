from itertools import chain

import torch

from openks.mm.graph import MMGraph, SemanticallySimilar

from .clip import *


class CLIP:
    def __init__(
        self,
        model_name: str = "ViT-B/32",
        jit: bool = True,
        device: str = "cpu",
        threshold: float = 20.0,
    ):
        self.device = torch.device(device)
        self.threshold = threshold
        self.model, self.preprocess = load(model_name, jit=jit, device=self.device)

    def __call__(self, graph: MMGraph) -> MMGraph:
        # TODO: support text entities
        entities = []
        features = []
        for entity in chain(
            graph.get_entities_by_concept("image"),
            graph.get_entities_by_concept("image_view"),
        ):
            image = self.preprocess(entity.image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.model.encode_image(image)
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )
            entity._clip_features = image_features.squeeze(0)

            entities.append(entity)
            features.append(image_features)

        # TODO: support large scale datasets
        with torch.no_grad():
            features = torch.cat(features, dim=0)
            logit_scale = self.model.logit_scale.exp()
            logits = logit_scale * features @ features.t()

        inds = (logits > self.threshold).nonzero(as_tuple=False)
        for ind1, ind2 in inds.tolist():
            if ind1 == ind2:
                continue
            graph.add_relation(
                SemanticallySimilar(
                    entities[ind1], entities[ind2], score=logits[ind1][ind2].item()
                )
            )

        return graph
