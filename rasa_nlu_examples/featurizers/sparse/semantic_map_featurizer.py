"""
This code has been commented away. It's a bit unclear to what extend we want
to continue supporting this feature, hence it was ignored during the 2.x -> 3.x port.
"""

# import json
# import re
# from collections import defaultdict
# from pathlib import Path
# from typing import Any, Dict, List, Optional, Set, Text, Tuple, Type, Union

# import numpy as np
# import scipy.sparse
# from rasa.nlu.components import Component
# from rasa.nlu.constants import (
#     DENSE_FEATURIZABLE_ATTRIBUTES,
#     FEATURIZER_CLASS_ALIAS,
#     TOKENS_NAMES,
# )
# from rasa.nlu.featurizers.featurizer import SparseFeaturizer
# from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
# from rasa.shared.nlu.constants import FEATURE_TYPE_SENTENCE, FEATURE_TYPE_SEQUENCE
# from rasa.shared.nlu.training_data.features import Features
# from rasa.shared.nlu.training_data.message import Message
# from rasa.shared.nlu.training_data.training_data import TrainingData


# class SemanticMapFeaturizer(SparseFeaturizer):
#     """Creates sparse features based on semantic map embeddings."""

#     @classmethod
#     def required_components(cls) -> List[Type[Component]]:
#         return [Tokenizer]

#     @classmethod
#     def required_packages(cls) -> List[Text]:
#         return []

#     defaults = {
#         # Filename of a pre-trained semantic map
#         "pretrained_semantic_map": None,
#         # How to combine sequence features to a sentence feature
#         "pooling": "sum",
#     }

#     def __init__(self, component_config: Optional[Dict[Text, Any]] = None) -> None:
#         """Constructs a new semantic map featurizer.

#         Args:
#             component_config: Configuration options.
#         """
#         super().__init__(component_config)
#         self.semantic_map = SemanticMap(self.pretrained_semantic_map_file)

#     def __eq__(self, other: Any) -> bool:
#         if not isinstance(other, type(self)):
#             return False
#         return self.semantic_map and (
#             self.semantic_map == other.semantic_map and self.pooling == other.pooling
#         )

#     def process(self, message: Message, **kwargs: Any) -> None:
#         """Adds semantic map features to a message.

#         Args:
#             message: The message to be processed.
#         """
#         for attribute in DENSE_FEATURIZABLE_ATTRIBUTES:
#             self._set_semantic_map_features(message, attribute)

#     def train(
#         self,
#         training_data: TrainingData,
#         *args: Any,
#         **kwargs: Any,
#     ) -> None:
#         """Processes all messages in the training data.

#         Args:
#             training_data: The NLU training data.
#         """
#         for example in training_data.training_examples:
#             self.process(example)

#     def _set_semantic_map_features(self, message: Message, attribute: Text) -> None:
#         """Adds semantic map features to the given attribute of the message.

#         Args:
#             message: The message to modify.
#             attribute: The name of the attribute that should be changed.
#         """
#         if not message.get(TOKENS_NAMES[attribute], []):
#             return

#         sequence_features, sentence_features = self._featurize_tokens(
#             message.get(TOKENS_NAMES[attribute], [])
#         )

#         if sequence_features is not None:
#             final_sequence_features = Features(
#                 sequence_features,
#                 FEATURE_TYPE_SEQUENCE,
#                 attribute,
#                 self.component_config[FEATURIZER_CLASS_ALIAS],
#             )
#             message.add_features(final_sequence_features)

#         if sentence_features is not None:
#             final_sentence_features = Features(
#                 sentence_features,
#                 FEATURE_TYPE_SENTENCE,
#                 attribute,
#                 self.component_config[FEATURIZER_CLASS_ALIAS],
#             )
#             message.add_features(final_sentence_features)

#     def _featurize_tokens(
#         self, tokens: List[Token]
#     ) -> Tuple[Optional[scipy.sparse.coo_matrix], Optional[scipy.sparse.coo_matrix]]:
#         """Featurizes the given tokens.

#         Args:
#             tokens: The tokens to be featurized.

#         Returns:
#             The sequence and sentence features.
#         """
#         if not tokens or not self.semantic_map:
#             return None, None

#         fingerprints = [
#             self.semantic_map.get_term_fingerprint(token.text) for token in tokens
#         ]

#         sequence_features = scipy.sparse.vstack(
#             [fp.as_coo_row_vector(append_oov_feature=True) for fp in fingerprints],
#             "coo",
#         )

#         if self.pooling == "merge":
#             sentence_features = self.semantic_map.merge(
#                 *fingerprints
#             ).as_coo_row_vector(append_oov_feature=True)
#         elif self.pooling == "mean":
#             sentence_features = scipy.sparse.coo_matrix(
#                 np.mean(sequence_features, axis=0)
#             )
#         elif self.pooling == "sum":
#             sentence_features = scipy.sparse.coo_matrix(
#                 np.sum(sequence_features, axis=0)
#             )
#         else:
#             raise ValueError(
#                 f"The 'pooling' option '{self.pooling}' must be one of 'sum' (default), 'mean', or 'merge'."
#             )

#         assert sequence_features.shape == (
#             len(fingerprints),
#             self.semantic_map.area + 1,
#         )
#         assert sentence_features.shape == (1, self.semantic_map.area + 1)

#         return sequence_features, sentence_features

#     @property
#     def pooling(self) -> Text:
#         """Determines the pooling operation."""
#         return self.component_config["pooling"]

#     @property
#     def pretrained_semantic_map_file(self) -> Optional[Path]:
#         """Returns the path of the pretrained semantic map, if any."""
#         filename = self.component_config.get("pretrained_semantic_map")
#         if filename and Path(filename).is_file():
#             return Path(filename)
#         else:
#             raise FileNotFoundError(
#                 f"Cannot find semantic map file '{filename}'. "
#                 f"Please check the 'pretrained_semantic_map' parameter."
#             )


# class SemanticFingerprint:
#     """Represents one particular semantic map embedding."""

#     def __init__(self, height: int, width: int, activations: Set[int]) -> None:
#         """Creates a new embedding.

#         Args:
#             height: Number of rows in the map.
#             width: Number of columns in the map.
#             activations: 1-D positions of active cells.
#         """
#         assert height > 0
#         assert width > 0
#         self.height = height
#         self.width = width
#         self.activations = activations

#     @property
#     def oov_feature(self) -> int:
#         """Gives the out-of-vocabulary feature."""
#         return 1 if len(self.activations) == 0 else 0

#     @property
#     def area(self):
#         """Gives the number of cells in this embedding."""
#         return self.height * self.width

#     def as_ascii_art(self, one_char="*", zero_char=" ") -> Text:
#         """Creates a string-image of this fingerprint.

#         Args:
#             one_char: Character to use for active cells.
#             zero_char: Character to use for inactive cells.

#         Returns:
#             A multi-line string that illustrates the sparse
#             binary matrix that is this fingerprint.
#         """
#         art = "\n"
#         for row in range(self.height):
#             for col in range(self.width):
#                 if col + self.width * row + 1 in self.activations:
#                     art += one_char
#                 else:
#                     art += zero_char
#             art += "\n"
#         return art

#     def as_activations(self) -> Set[int]:
#         """Gives the active cell indices."""
#         return self.activations

#     def as_csr_matrix(self) -> scipy.sparse.csr_matrix:
#         """Gives the fingerprint as csr_matrix."""
#         data = np.ones(len(self.activations))
#         row_indices = [(a - 1) // self.width for a in self.activations]
#         col_indices = [(a - 1) % self.width for a in self.activations]

#         return scipy.sparse.csr_matrix(
#             (data, (row_indices, col_indices)),
#             shape=(self.height, self.width),
#             dtype=np.float32,
#         )

#     def as_coo_row_vector(
#         self, append_oov_feature: bool = False
#     ) -> scipy.sparse.coo_matrix:
#         """Gives the fingerprint as sparse vector."""
#         fingerprint_features = self.as_csr_matrix().reshape((1, -1))
#         if append_oov_feature:
#             return scipy.sparse.hstack(
#                 [fingerprint_features, [self.oov_feature]]
#             ).tocoo()
#         else:
#             return fingerprint_features.tocoo()

#     def as_dense_vector(self) -> np.array:
#         """Gives the fingerprint as dense vector."""
#         return np.reshape(self.as_csr_matrix().todense(), (self.height * self.width,))


# class SemanticMap:
#     """Represents a collection of semantic map embeddings (a.k.a. fingerprints)."""

#     def __init__(self, filename: Union[Path, Text]) -> None:
#         """Creates a new semantic map.

#         Args:
#             filename: File to load the map from.
#         """
#         if not filename or not Path(filename).is_file:
#             raise FileNotFoundError(f"Cannot find semantic map file '{filename}'")

#         with open(filename, "r", encoding="utf-8") as file:
#             _data = json.load(file)

#         try:
#             self._width = _data["Width"]
#             self._height = _data["Height"]
#             self._local_topology = _data["LocalTopology"]
#             self._global_topology = _data["GlobalTopology"]
#             self._assume_lower_case = _data.get("AssumeLowerCase")
#             self._max_number_of_active_cells = _data.get("MaximumNumberOfActiveCells")
#             self._embeddings: Dict[Text, List[int]] = _data["Embeddings"]
#         except KeyError as error:
#             raise ValueError(f"Semantic map file '{filename}' misses a key: {error}.")

#         self._prepare_vocab_patterns()

#     def _prepare_vocab_patterns(self) -> None:
#         """Prepares regex patterns used to find known words in a string."""
#         self._vocab_pattern = re.compile(
#             "|".join(
#                 [
#                     r"\b" + re.escape(word) + r"\b"
#                     for word in self._embeddings.keys()
#                     if not word.startswith("<")
#                 ]
#             )
#         )

#     def __eq__(self, other: Any) -> bool:
#         return (
#             self.width == other.width
#             and self.height == other.height
#             and self.local_topology == other.local_topology
#             and self.global_topology == other.global_topology
#             and self.vocabulary == other.vocabulary
#             and self._assume_lower_case == other._assume_lower_case
#             and all(
#                 self._embeddings.get(token) == other._embeddings.get(token)
#                 for token in self.vocabulary
#             )
#         )

#     def get_empty_fingerprint(self) -> SemanticFingerprint:
#         """Generate an empty embedding."""
#         return SemanticFingerprint(self.height, self.width, set())

#     def get_term_activations(self, term: Text) -> List[int]:
#         """Get the active cells for the given term.

#         Args:
#             term: The word in question.

#         Returns:
#             The 1D-indices of the active cells.
#         """
#         if self._assume_lower_case:
#             return self._embeddings.get(term.lower())
#         else:
#             activations = self._embeddings.get(term)
#             if not activations and self._assume_lower_case is None:
#                 # Try the lower-case version if we don't make assumptions
#                 # and normal case didn't work
#                 return self._embeddings.get(term.lower())
#             else:
#                 return activations

#     def get_term_fingerprint(self, term: Text) -> SemanticFingerprint:
#         """Get the embedding for the given term.

#         Args:
#             term: The word in question.

#         Returns:
#             The semantic map embedding of the term.
#         """
#         activations = self.get_term_activations(term)
#         if not activations:
#             return self.get_empty_fingerprint()
#         else:
#             return SemanticFingerprint(self.height, self.width, set(activations))

#     def get_fingerprint(self, text: Text) -> SemanticFingerprint:
#         """Get the embedding for the given text.

#         Args:
#             text: The text in question. This might be one word,
#                   a sentence, or an entire document.

#         Returns:
#             The semantic map embedding of the text.
#         """
#         term_fingerprints = [
#             self.get_term_fingerprint(term) for term in self.get_known_terms(text)
#         ]
#         if term_fingerprints:
#             return self.merge(*term_fingerprints)
#         else:
#             return self.get_empty_fingerprint()

#     def get_known_terms(self, text: Text) -> List[Text]:
#         """Extracts words from text that are in the vocabulary.

#         Args:
#             text: The text to extract from.

#         Returns:
#             The vocabulary words that the text is made of (in order
#             of appearance and with duplicates).
#         """
#         _text = text.lower() if self._assume_lower_case else text
#         terms = self._vocab_pattern.findall(_text)
#         return terms

#     def merge(
#         self, *fingerprints: SemanticFingerprint, boost_factor=0.618033988749
#     ) -> SemanticFingerprint:
#         """Perform a symmetric semantic merge operation on the embeddings.

#         Args:
#             fingerprints: The embeddings to be merged.
#             boost_factor: How much neighbouring cells should strenthen each other.

#         Returns:
#             The merged embedding.
#         """
#         if fingerprints:
#             # Count how often which cell is active
#             activation_count = defaultdict(int)
#             for fingerprint in fingerprints:
#                 for activation in fingerprint.as_activations():
#                     activation_count[activation] += 1

#             # Boost the counts by their neighbours
#             def total_neighbouring_activation_count(cell: int) -> int:
#                 return sum(
#                     [
#                         activation_count[neighbour]
#                         for neighbour in _neightbours_iterator(
#                             cell,
#                             self.height,
#                             self.width,
#                             self.local_topology,
#                             self.global_topology,
#                         )
#                         if neighbour in activation_count
#                     ]
#                 )

#             boosted_activation_count = {
#                 cell: n + boost_factor * total_neighbouring_activation_count(cell)
#                 for cell, n in activation_count.items()
#             }

#             # Drop all but the strongest activations
#             all_activations = [
#                 cell
#                 for cell, strength in sorted(
#                     boosted_activation_count.items(),
#                     key=(lambda item: item[1]),
#                     reverse=True,
#                 )
#             ]
#             activations = (
#                 all_activations[: self.max_number_of_active_cells]
#                 if self.max_number_of_active_cells < len(all_activations)
#                 else all_activations
#             )

#             return SemanticFingerprint(self.height, self.width, set(activations))
#         else:
#             return self.get_empty_fingerprint()

#     def is_vocabulary_member(self, term: Text) -> bool:
#         """Checks whether the given term is in the vocabulary.

#         Args:
#             term: The query term.

#         Returns:
#             True if and only if the term is in the vocabulary.
#         """
#         return term in self._embeddings

#     def has_fingerprint(self, text: Text) -> bool:
#         """Checks if the given text has a non-empty embedding.

#         Args:
#             text: The query text.

#         Returns:
#             True if and only if the given text would yield a non-empty fingerprint.
#         """
#         return len(self.get_known_terms(text)) > 0

#     def as_dict(self) -> Dict[Text, Any]:
#         """Converts the semantic map into a dict.

#         Returns:
#             A dict with the defining properties of this map, including
#             all stored word embeddings.
#         """
#         return {
#             "Width": self.width,
#             "Height": self.height,
#             "LocalTopology": self.local_topology,
#             "GlobalTopology": self.global_topology,
#             "AssumeLowerCase": self._assume_lower_case,
#             "Embeddings": self._embeddings,
#         }

#     @property
#     def vocabulary(self) -> Set[Text]:
#         """Gives the vocabulary of words that have embeddings."""
#         if not self._embeddings:
#             return set()
#         return set(self._embeddings.keys())

#     @property
#     def width(self) -> int:
#         """Gives the number of columns of the map."""
#         return self._width

#     @property
#     def height(self) -> int:
#         """Gives the number of rows in the map."""
#         return self._height

#     @property
#     def area(self):
#         """Gives the number of cells in an embedding."""
#         return self.height * self.width

#     @property
#     def local_topology(self) -> int:
#         """Gives the number of neighbours that a typical cell has.

#         Returns:
#             The number of neighbours that a non-edge cell has. This number
#             corresponds to the distance metric used during map creation:
#             4: Manhattan distance (four neighbours per cell)
#             6: Hexagonal distance (six neighbours per cell)
#             8: Chessboard distance (eight neibours per cell)
#         """
#         return self._local_topology

#     @property
#     def global_topology(self) -> int:
#         """Gives the number of edges that the map has.

#         Returns:
#             0: Torus - the east/west and north/south edges are identical
#             1: Moebius - the east edge is identical to the fliped west edge
#             2: Cylinder - the east/west edges are identical
#             4: Rectangle - all four edges are distinct
#         """
#         num_edges = {"torus": 0, "moebius": 1, "cylinder": 2, "rectangle": 4}.get(
#             self._global_topology
#         )
#         if num_edges is None:
#             raise ValueError(f"Unknown global topology `{self._global_topology}`.")
#         return num_edges

#     @property
#     def max_number_of_active_cells(self) -> int:
#         """Gives the maximum number of active cells in any embedding."""
#         if not self._max_number_of_active_cells:
#             return np.math.ceil(self.area * 0.02)
#         else:
#             return self._max_number_of_active_cells


# def _neightbours_iterator(
#     cell: int,
#     height: int,
#     width: int,
#     local_topology: int = 8,
#     global_topology: int = 0,
# ) -> callable:
#     """Yields the neighbours of the cell position, depending on topology.

#     Args:
#         cell: Index of the cell in question.
#         height: Number of rows in the semantic map.
#         width: Number of columns in the semantic map.
#         local_topology: Number of neighbours or a central cell.
#         global_topology: Number of edges of the map.

#     Yields:
#         The index of a neighbouring cell.
#     """
#     if global_topology == 0:
#         _shift_onto_map = _shift_onto_map_torus
#     else:
#         raise NotImplementedError(
#             "Merge operation for non-torus maps is not yet implemented"
#         )
#     if local_topology == 4:
#         yield _shift_onto_map(cell - 1, height, width),  # Left
#         yield _shift_onto_map(cell + 1, height, width),  # Right
#         yield _shift_onto_map(cell - width, height, width),  # Top
#         yield _shift_onto_map(cell + width, height, width),  # Bottom
#     elif local_topology == 6:
#         x = (cell - 1) % width
#         y = (cell - 1) // width
#         if y // 2 == 0:
#             yield _shift_onto_map(
#                 (x - 1) + (y - 1) * width + 1, height, width
#             ),  # Top left
#             yield _shift_onto_map(x + (y - 1) * width + 1, height, width),  # Top right
#             yield _shift_onto_map((x - 1) + y * width + 1, height, width),  # Left
#             yield _shift_onto_map((x + 1) + y * width + 1, height, width),  # Right
#             yield _shift_onto_map(
#                 (x - 1) + (y + 1) * width + 1, height, width
#             ),  # Bottom left
#             yield _shift_onto_map(
#                 x + (y + 1) * width + 1, height, width
#             ),  # Bottom right
#         else:
#             yield _shift_onto_map(x + (y - 1) * width + 1, height, width),  # Top left
#             yield _shift_onto_map(
#                 (x + 1) + (y - 1) * width + 1, height, width
#             ),  # Top right
#             yield _shift_onto_map((x - 1) + y * width + 1, height, width),  # Left
#             yield _shift_onto_map((x + 1) + y * width + 1, height, width),  # Right
#             yield _shift_onto_map(
#                 x + (y + 1) * width + 1, height, width
#             ),  # Bottom left
#             yield _shift_onto_map(
#                 (x + 1) + (y + 1) * width + 1, height, width
#             ),  # Bottom right
#     elif local_topology == 8:
#         yield _shift_onto_map(cell - 1, height, width),  # Left
#         yield _shift_onto_map(cell + 1, height, width),  # Right
#         yield _shift_onto_map(cell - width, height, width),  # Top
#         yield _shift_onto_map(cell + width, height, width),  # Bottom
#         yield _shift_onto_map(cell - 1 - width, height, width),  # Top Left
#         yield _shift_onto_map(cell + 1 - width, height, width),  # Top Right
#         yield _shift_onto_map(cell - 1 + width, height, width),  # Bottom Left
#         yield _shift_onto_map(cell + 1 + width, height, width),  # Bottom Right
#     else:
#         raise ValueError("Local topology must be either 4, 6, or 8.")


# def _shift_onto_map_torus(cell: int, height: int, width: int) -> int:
#     """Ensures that the given cell is on the map by translating its position.

#     Args:
#         Position of the cell.

#     Returns:
#         Shifted position of the cell.
#     """
#     # Globally the map's topology is a torus, so
#     # top and bottom edges are connected, and left
#     # and right edges are connected.
#     x = (cell - 1) % width
#     y = (cell - 1) // width
#     if y < 0:
#         y += height * abs(y // height)
#     return x + y * width + 1


# def semantic_overlap(
#     fp1: SemanticFingerprint, fp2: SemanticFingerprint, method: Text = "Jaccard"
# ) -> float:
#     """Computes the overlap score of the two fingerprints.

#     The score is a floating point number between 0 and 1, where
#     0 means that the two words are unrelated and 1 means that
#     they share exactly the same meaning.

#     Args:
#         fp1: First semantic fingerprint.
#         fp2: Semantic fingerprint to compare with (the order is irrelevant).
#         method: The method to use for comparison. This can be one of
#                 "SzymkiewiczSimpson", "Jaccard", or "Rand".
#     """
#     if method == "SzymkiewiczSimpson":
#         return _szymkiewicz_simpson_overlap(fp1, fp2)
#     elif method == "Jaccard":
#         return _jaccard_overlap(fp1, fp2)
#     elif method == "Rand":
#         return _rand_overlap(fp1, fp2)
#     else:
#         raise ValueError(
#             f"Method '{method}' is not one of 'SzymkiewiczSimpson', 'Jaccard', or 'Rand'"
#         )


# def _szymkiewicz_simpson_overlap(
#     fp1: SemanticFingerprint, fp2: SemanticFingerprint
# ) -> float:
#     num_common = len(fp1.as_activations().intersection(fp2.as_activations()))
#     min_length = min(len(fp1.as_activations()), len(fp2.as_activations()))
#     if min_length == 0:
#         return 0
#     else:
#         return float(num_common / min_length)


# def _jaccard_overlap(fp1: SemanticFingerprint, fp2: SemanticFingerprint) -> float:
#     num_common = len(fp1.as_activations().intersection(fp2.as_activations()))
#     union_length = len(fp1.as_activations().union(fp2.as_activations()))
#     if union_length == 0:
#         return 1.0
#     else:
#         return float(num_common / union_length)


# def _rand_overlap(fp1: SemanticFingerprint, fp2: SemanticFingerprint) -> float:
#     num_cells = fp1.height * fp1.width
#     num_11 = len(fp1.as_activations().intersection(fp2.as_activations()))
#     num_10 = len(fp1.as_activations().difference(fp2.as_activations()))
#     num_01 = len(fp2.as_activations().difference(fp1.as_activations()))
#     num_00 = num_cells - (num_10 + num_01 + num_11)
#     return float((num_00 + num_11) / num_cells)
