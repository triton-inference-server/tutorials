# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import itertools
from dataclasses import dataclass
from typing import Any, Dict, Hashable, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from theine import Cache


class KeyMapper:
    """
    A class to manage bidirectional mapping between hashable keys and integer IDs.
    """

    def __init__(self):
        self.hk_map: Dict[Hashable, int] = {}
        self.kh_map: Dict[int, Hashable] = {}
        self.counter = itertools.count()

    def add_key(self, key: Hashable):
        """
        Add a new key to the mapper and return its assigned ID.

        Args:
            key (Hashable): The key to be added.

        Returns:
            int: The assigned ID for the key.
        """
        if key in self.hk_map.keys():
            return None
        id = next(self.counter)
        self.hk_map[key] = id
        self.kh_map[id] = key
        return id

    def remove_key(self, key: Hashable):
        """
        Remove key from the mapper and return its ID.

        Args:
            key (Hashable): The key to be removed.

        Returns:
            int: The ID for the removed key.
        """
        id = self.hk_map.pop(key, None)
        if id is not None:
            self.kh_map.pop(id, None)
            return id
        return None

    def get_key(self, id: int):
        """
        Retrieve the key associated with the given ID.

        Args:
            id (int): The ID to look up.

        Returns:
            Optional[Hashable]: The associated key, or None if not found.
        """
        return self.kh_map.get(id)

    def get_id(self, key: Hashable):
        """
        Retrieve the ID associated with the given key.

        Args:
            key (Hashable): The key to look up.

        Returns:
            Optional[int]: The associated ID, or None if not found.
        """
        return self.hk_map.get(key)


@dataclass
class SemanticCPUCacheConfig:
    """
    Configuration class for SemanticCPUCache.

    Attributes:
        cache (Any): The cache object to use.
            Default: Cache(policy="lru", size=1000).
        encoder (Any): The encoder object for embedding queries.
            Default: SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        encoder_dim (int): The encoder dimension.
            Default: 384. The size of `all-MiniLM-L6-v2` embeddings.
        index (Any): The index object for similarity search.
            Default: faiss.IndexIDMap(faiss.IndexFlatL2(encoder_dim))
        threshold (float): The similarity threshold for considering a match.
            Default: 0.25
        key_mapper (Any): The key mapper object for managing key-ID mappings.
            default: KeyMapper()
    """

    cache: Any = Cache(policy="lru", size=1000)
    encoder: Any = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    encoder_dim: int = 384
    index: Any = faiss.IndexIDMap(faiss.IndexFlatL2(encoder_dim))
    threshold: float = 0.25
    key_mapper: Any = KeyMapper()


class SemanticCPUCache:
    """
    Semantic cache implementation.
    """

    def __init__(self, config: SemanticCPUCacheConfig):
        """
        Initialize the SemanticCPUCache with the given configuration.

        Args:
            config (SemanticCPUCacheConfig): The configuration object.
        """
        self.encoder = config.encoder
        self.index = config.index
        self.cache = config.cache
        self.key_map = config.key_mapper
        self.threshold = config.threshold

    def get(self, key: Hashable, default: Any = None) -> Any:
        """
        Retrieve a value from the cache based on the given key.

        First, a similarity search is performed. If a similar key is found
        within the threshold, its associated value is returned.
        Otherwise, the default value is returned.

        Args:
            key (Hashable): The key to look up.
            default (Any, optional): The default value to return if no match is found. Defaults to None.

        Returns:
            Any: The retrieved value or the default value.
        """
        if self.index.ntotal < 1:
            return default

        key_search = np.asarray([self.encoder.encode(key)])
        # The vector index returns two values, distance to the most similar
        # embedding (1 indicates we only need top 1 similar result), and
        # its numerical index.
        dist, ind = self.index.search(key_search, 1)

        # If the distance between vectors above the set threshold, i.e.
        # the most similar embedding is too far from the current prompt
        # embedding, this considered as cache miss and we return the `default`.
        if dist[0][0] > self.threshold:
            return default

        # To retrieve the cache hit from the cache store, we need to retrieve
        # the corresponding prompt from the key_map store, given its index.
        key_str = self.key_map.get_key(ind[0][0])

        return self.cache.get(key=key_str, default=default)

    def set(self, key: Hashable, value: Any) -> Optional[str]:
        """
        Set a key-value pair in the cache.

        This method adds the key to the key mapper, encodes the key,
        adds the encoded key to the index, and sets the value in the cache.

        Args:
            key (Hashable): The key to set.
            value (Any): The value to associate with the key.

        Returns:
            Optional[str]: The result of setting the value in the cache.

        Raises:
            AssertionError: If the key could not be added to the key mapper.
        """
        id = self.key_map.add_key(key)
        assert id is not None, "Adding key to the key map failed, returned id is None."
        self.index.add_with_ids(
            np.expand_dims(self.encoder.encode(key), axis=0), np.asarray([id])
        )
        # Adding a new entry into the cache can evict an old entry, according
        # to the policy in-use. We need to make sure we evict the same entry
        # from the vector index, stored in `self.index`.
        evicted_key = self.cache.set(key, value)
        self._handle_evicted_key(evicted_key=evicted_key)

        return None

    def _handle_evicted_key(self, evicted_key: Optional[Hashable]) -> None:
        """
        Handle the eviction of a key from the cache.

        This method is called when a key is evicted from the cache. It removes
        the evicted key from the key_map and its corresponding
        vector embedding from the index.

        Args:
            evicted_key (Optional[Hashable]): The key that was evicted from the
            cache.
        """
        if evicted_key:
            evicted_id = self.key_map.remove_key(evicted_key)
            self.index.remove_ids(np.array([evicted_id]))
        return None
