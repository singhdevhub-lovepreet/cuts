from __future__ import annotations

from pathlib import Path

from cuts.graph import Context, Node, Pipeline


class FirstNode(Node):
    name = "first"
    provides = ("alpha",)

    def run(self, context: Context) -> Context:
        context.extras["first"] = True
        return context


class SecondNode(Node):
    name = "second"
    requires = ("alpha",)
    provides = ("beta",)

    def run(self, context: Context) -> Context:
        context.extras["second"] = True
        return context


class ThirdNode(Node):
    name = "third"
    requires = ("beta",)

    def run(self, context: Context) -> Context:
        context.extras["third"] = True
        return context


def test_pipeline_orders_by_dependencies() -> None:
    pipeline = Pipeline([ThirdNode(), FirstNode(), SecondNode()])
    assert [node.name for node in pipeline.ordered_nodes] == ["first", "second", "third"]
    context = Context(source_paths=(Path("clip.mp4"),))
    result = pipeline.run(context)
    assert result.extras == {"first": True, "second": True, "third": True}
