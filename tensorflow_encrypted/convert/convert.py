"""Convert module helps convert pre-trained tensorflow models to a usable format."""


from typing import Dict, Tuple, List, Any, Union
import tensorflow as tf
from collections import Iterable
import numpy as np

from ..io import InputProvider
from ..player import Player
from ..protocol.protocol import Protocol
from ..config import Config


class ConvertInputProvider(InputProvider):
    """
    ConvertInputProvider is a helper InputProvider.

    This class is intended to be passed into the `convert()` function.
    """

    input: np.ndarray

    def __init__(self, player: Player, input: np.ndarray) -> None:
        """
        Initialize ConvertInputProvider.

        Args:
            player: `Player`: the player whose providing the input
            input: `np.ndarray`: the actual input
        """
        self.input = input
        self.player = player

    def provide_input(self) -> tf.Tensor:
        """
        Implement the this function of the base class.

        Returns:
            The input wrapped in a constant.

        """
        return tf.constant(self.input)


class Converter():
    """
    A converter handles converting pre-trained tensorflow graph into a graph usabled by tf-encrypted.

    Uses a `Config` and `Protocol` to generate the correct graph for the given `tf.GraphDef`.
    """

    config: Config
    protocol: Protocol
    outputs: Dict[str, Any] = {}
    weights_provider: Player

    def __init__(self, config: Config, protocol: Protocol,
                 weights_provider: Player) -> None:
        """
        Initialize Converter.

        Args:
            config: `Config`: the server's involved configuration
            protocol: `Protocol`: the protocol to use when converting graph
            weights_provider: `Player`: the player whose going to provider the weights
        """
        self.config = config
        self.protocol = protocol
        self.weights_provider = weights_provider

    def convert(self, graph_def: tf.GraphDef, input: Union[List[InputProvider], InputProvider],
                registry: Dict[str, Any]) -> tf.Tensor:
        """
        Convert a pre-trained tensorflow graph into a graph usable by tf-encrypted.

        Args:
          graph_def: `tf.GraphDef`: tensorflow graph protobuf representation
          input: `Union[List[InputProvider]: InputProvider]`: inputs into the graph
          registry: `Dict[str: Any]`: the transform function registry

        Returns:
            The output `tf.Tensor` in the graph.

        """
        name_to_input_name, name_to_node = extract_graph_summary(graph_def)

        if isinstance(input, InputProvider):
            i = [input]
        elif isinstance(input, Iterable):
            i = input
        else:
            i = []

        iter = enumerate(i)

        for output, inputs in name_to_input_name.items():
            node = name_to_node[output]

            if node.op == "Placeholder":
                try:
                    count, item = iter.__next__()
                except StopIteration:
                    raise InvalidArgumentError("Not enough placeholders supplied")

                x = self.protocol.define_private_input(item)

                self.outputs[output] = x
                continue

            self.outputs[output] = registry[node.op](self, node, inputs)

        return self.outputs[graph_def.node[-1].name]


def node_name(n: str) -> str:
    """
    Remove unneeded characters from the node name `str`.

    Args:
      n: `str`: the node name

    Returns:
        Cleaned up node name as a `str`.

    """
    if n.startswith("^"):
        return n[1:]
    else:
        return n.split(":")[0]


def extract_graph_summary(graph_def: tf.GraphDef) -> Tuple[Dict[str, List[str]],
                                                           Dict[str, Any]]:
    """
    Extract useful information from the graph and returns them.

    Args:
      graph_def: `tf.GraphDef`:

    Returns:
        A dictionary that maps node names to all of the names of the input nodes to the dest node.
        A dictionary that maps node names to the actual node.

    """
    name_to_input_name = {}  # Keyed by the dest node name.
    name_to_node = {}  # Keyed by node name.

    for node in graph_def.node:
        name: str = node.name
        n = node_name(name)
        name_to_node[n] = node
        name_to_input_name[n] = [node_name(x) for x in node.input]

    return name_to_input_name, name_to_node


class InvalidArgumentError(Exception):
    """Invalid arg passed into a convert function."""

    pass
