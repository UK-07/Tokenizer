# -*- coding: utf-8 -*-
"""Tokenizer class to create a byte-pair encoding vocabulary."""
from typing import Dict, List, Tuple


class Tokenizer:
  """Tokenizer class to train a vocabulary, encode str and decode tokens."""

  def __init__(self):
    """Initialize a tokenizer of given `vocab_size`."""
    self.pair_to_token = {}
    self.token_to_pair = {}
    # Mapping from token to bytearray. Bytearray is mutable and allows faster
    # and inplace merges.
    self.vocab = {}

  def update_pair(
      self, tokens: List[int], pair: Tuple[int, int], swap: int) -> List[int]:
    """Swap every occurance of `pair` in `tokens` with `swap`.

    Args:
      tokens: List of integers representing tokens in the vocabulary.
      pair: Pair of integers, tokens which should be swapped.
      swap: Int, token to swap the pair with.

    Returns:
      Copy of `tokens` after swapping.
    """
    tokens_swapped = []
    found_pair = False

    for i in range(len(tokens) - 1):
      if found_pair:
        found_pair = False
        continue

      if (tokens[i], tokens[i + 1]) == pair:
        found_pair = True
        tokens_swapped.append(swap)
      else:
        tokens_swapped.append(tokens[i])

    if not found_pair:
      tokens_swapped.append(tokens[-1])

    return tokens_swapped

  def count_pairs(self, tokens: List[int]) -> Dict[Tuple[int], int]:
    """Returns a mapping from token pairs to their frequency in `tokens`.

    Args:
      tokens: List of integers representing tokens in the vocabulary.

    Returns:
      Mapping from sub-sequent pairs to their frequency in `tokens`.
    """
    pair_count = {}
    for f, s in zip(tokens, tokens[1:]):
      pair = (f, s)
      pair_count[pair] = pair_count.get(pair, 0) + 1

    return pair_count

  def tokenize(self, text: str):
    """Tokenize the given `text` string.

    This can be overridden in decendant classes for differnt tokenization
    strategies.

    Args:
      text: The text to tokenize.

    Returns:
      List of integers representing tokens and the next unmapped token.
    """
    # Token mapping: UTF-8 string -> bytes -> ints
    tokens = list(map(int, text.encode('utf-8')))
    if not self.vocab:
      self.vocab = {i: bytearray([i]) for i in range(256)}
    next_token = 256
    return tokens, next_token

  def build(self, text: str, vocab_size: int):
    """Create a byte-pair encoding vocabulary from the given `text` string.

    Token pairs are merged till the specified vocab_size is reached.

    Args:
      text: The text to train the tokenizer on.
      vocab_size: The maximum size of the vocabulary.
    """
    tokens, next_token = self.tokenize(text)

    while next_token <= vocab_size:
      # Recompute and fetch pair with greatest frequency.
      pair_count = self.count_pairs(tokens)
      f, s = max(pair_count, key=pair_count.get)

      # Update token maps and add new token to vocabulary.
      self.pair_to_token[(f, s)] = next_token
      self.token_to_pair[next_token] = (f, s)
      self.vocab[next_token] = self.vocab[f] + self.vocab[s]

      # Swap all occurances of chosen pair with new token.
      tokens = self.update_pair(tokens, (f, s), next_token)

      next_token += 1

  def decode(self, tokens: List[int]) -> str:
    """Given a list of tokens, return the `utf-8` string."""
    byte_string = b''.join([self.vocab[token] for token in tokens])
    # Replace ensures invalid token sequence prediction doesn't cause errors.
    return byte_string.decode('utf-8', errors='replace')

  def encode(self, text: str) -> List[int]:
    """Encode the given text into learnt byte-pair encoding.

    Args:
      text: The text to encode.

    Returns:
      List of integers representing tokens in the learnt encoding.
    """
    tokens, _ = self.tokenize(text)

    # Swap the pairs with the smallest mapping first.
    for token in sorted(self.token_to_pair.keys()):
      pair = self.token_to_pair[token]
      tokens = self.update_pair(tokens, pair, token)

    return tokens
