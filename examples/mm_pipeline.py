from openks.mm.pipeline import Pipeline
from openks.mm.data_loader import GlobDataLoader


def main():
    pipeline = Pipeline(
        GlobDataLoader(root_dir="../datasets/images"),
    )
    graph = pipeline.run()
    graph.save("../datasets/test-kg")


if __name__ == "__main__":
    main()
