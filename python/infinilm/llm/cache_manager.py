"""
KV Cache Manager - Paged Attention block-based cache allocation and management.
"""

from collections import deque
from typing import List, Dict, Set
import xxhash
import numpy as np


class Block:
    """KV Cache Block with reference counting and hash-based reuse support."""

    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids: List[int] = []
        self.computed_tokens = 0
        self.committed = False

    def update(
        self,
        hash_value: int,
        token_ids: List[int],
        committed: bool = False,
    ) -> None:
        self.hash = hash_value
        self.token_ids = token_ids.copy()
        self.computed_tokens = len(token_ids)
        self.committed = committed

    def reset(self) -> None:
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []
        self.computed_tokens = 0
        self.committed = False

    def free(self) -> None:
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []
        self.computed_tokens = 0
        self.committed = False

    def __repr__(self) -> str:
        return (
            f"Block(id={self.block_id}, ref={self.ref_count}, hash={self.hash}, "
            f"computed={self.computed_tokens}, committed={self.committed})"
        )


class BlockManager:
    """Manages Paged KV Cache allocation with prefix caching support.

    Features:
    - Block allocation/deallocation with reference counting
    - Hash-based prefix caching for token sequence reuse
    - Slot mapping generation for physical-to-logical position mapping
    """

    def __init__(self, num_blocks: int, block_size: int):
        assert (
            num_blocks > 0 and block_size > 0
        ), "num_blocks and block_size must be positive"
        self.num_blocks = num_blocks
        self.block_size = block_size

        self.blocks: List[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: Dict[int, int] = {}
        self.free_block_ids: deque = deque(range(num_blocks))
        self.used_block_ids: Set[int] = set()
        self.req_block_ids: Set[int] = set()

    def reset_req_blocks(self) -> None:
        """Move blocks from prefill stage to used blocks and update hash mappings."""
        for block_id in self.req_block_ids:
            self.used_block_ids.add(block_id)
            block = self.blocks[block_id]
            prefix_hash = block.hash
            if prefix_hash != -1:
                block.committed = True
                self.hash_to_block_id[prefix_hash] = block_id
        self.req_block_ids.clear()

    @classmethod
    def compute_hash(cls, token_ids: List[int], prefix_hash: int = -1) -> int:
        """Compute hash for token sequence with optional prefix chaining."""
        h = xxhash.xxh64()
        if prefix_hash != -1:
            h.update(prefix_hash.to_bytes(8, "little"))
        h.update(np.array(token_ids, dtype=np.int32).tobytes())
        return h.intdigest()

    def _allocate_partial_block(self, block_id: int) -> Block:
        """Allocate an incomplete block and add to used blocks."""
        assert block_id in self.free_block_ids, f"Block {block_id} not in free list"
        block = self.blocks[block_id]
        assert block.ref_count == 0, f"Block {block_id} ref_count not zero"

        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return block

    def _allocate_full_block(self, block_id: int) -> Block:
        """Allocate a complete block and add to request blocks."""
        assert block_id in self.free_block_ids, f"Block {block_id} not in free list"
        block = self.blocks[block_id]
        assert block.ref_count == 0, f"Block {block_id} ref_count not zero"

        block.reset()
        self.free_block_ids.remove(block_id)
        self.req_block_ids.add(block_id)
        return block

    def _deallocate_block(self, block_id: int):
        """Deallocate a block and return it to free list."""
        block = self.blocks[block_id]
        assert (
            block.ref_count == 0
        ), f"Block {block_id} ref_count not zero, cannot deallocate"

        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]

        block.free()
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def _mark_block_computed(
        self,
        block_id: int,
        block_start: int,
        computed_start: int,
        computed_end: int,
    ) -> None:
        """Mark a contiguous token prefix in one block as computed."""
        block = self.blocks[block_id]
        local_start = max(computed_start - block_start, 0)
        local_end = min(computed_end - block_start, self.block_size)
        if local_end <= local_start:
            return
        if local_start > block.computed_tokens:
            raise ValueError("prefill chunk leaves an uncomputed block gap")
        block.computed_tokens = max(block.computed_tokens, local_end)

    def _commit_block(
        self,
        block_id: int,
        block_tokens: List[int],
        prefix_hash: int,
    ) -> int:
        """Commit a fully computed block into the prefix cache."""
        if len(block_tokens) != self.block_size:
            raise ValueError("only full blocks can be committed")

        block = self.blocks[block_id]
        if block.computed_tokens != self.block_size:
            raise ValueError("cannot commit a block before it is fully computed")

        current_hash = self.compute_hash(block_tokens, prefix_hash)
        if block.committed:
            if block.hash != current_hash or block.token_ids != block_tokens:
                raise ValueError("committed block hash/token mismatch")
            return block.hash

        block.update(current_hash, block_tokens, committed=True)
        self.hash_to_block_id[current_hash] = block_id
        return current_hash

    def can_allocate(self, num_required_blocks: int) -> bool:
        return len(self.free_block_ids) >= num_required_blocks

    def allocate_prefill_chunk(
        self,
        block_table: List[int],
        start: int,
        end: int,
    ) -> tuple[List[int], List[int]]:
        """Allocate blocks and slots for a single prefill chunk.

        Chunked prefill computes only ``[start, end)`` in the current forward.
        This method only returns slot mappings for that range and deliberately
        avoids prefix-cache hash commits for blocks that may not be fully
        computed yet.
        """
        if block_table is None:
            block_table = []
        if start < 0:
            raise ValueError("prefill chunk start must be non-negative")
        if end < start:
            raise ValueError("prefill chunk end must be greater than or equal to start")

        required_blocks = (end + self.block_size - 1) // self.block_size
        missing_blocks = required_blocks - len(block_table)

        if missing_blocks > 0:
            if not self.can_allocate(missing_blocks):
                if not self.try_free_blocks(missing_blocks):
                    raise RuntimeError("No available cache blocks for prefill chunk")

            for _ in range(missing_blocks):
                block_id = self.free_block_ids[0]
                self._allocate_partial_block(block_id)
                block_table.append(block_id)

        slot_mapping = self._build_slot_mapping(block_table, start, end)
        return block_table, slot_mapping

    def commit_computed_blocks(
        self,
        block_table: List[int],
        token_ids: List[int],
        computed_start: int,
        computed_end: int,
        committed_until: int,
        prefix_hash: int = -1,
    ) -> tuple[int, int]:
        """Commit full blocks after a forward-computed token range succeeds.

        Args:
            block_table: Physical blocks for the request sequence.
            token_ids: Complete token sequence used for prefix-cache hashes.
            computed_start: First token position written by the completed forward.
            computed_end: Exclusive token position written by the completed forward.
            committed_until: First uncommitted token position. This must be on a
                block boundary and commits advance contiguously from here.
            prefix_hash: Prefix hash at committed_until. Use -1 for no prefix.

        Returns:
            Tuple of (new_committed_until, new_prefix_hash).

        This method is the precise post-forward commit boundary: allocation may
        happen earlier, but prefix-cache visibility only changes here.
        """
        if block_table is None:
            raise ValueError("block_table cannot be None")
        if computed_start < 0 or computed_end < computed_start:
            raise ValueError("invalid computed token range")
        if computed_end > len(token_ids):
            raise ValueError("computed token end cannot exceed token length")
        if committed_until < 0 or committed_until > computed_end:
            raise ValueError("invalid committed token progress")
        if committed_until % self.block_size != 0:
            raise ValueError("committed token progress must align to block boundary")

        required_blocks = (computed_end + self.block_size - 1) // self.block_size
        if len(block_table) < required_blocks:
            raise RuntimeError("block_table does not cover committed token range")

        if committed_until > 0 and prefix_hash == -1:
            prev_block = self.blocks[
                block_table[committed_until // self.block_size - 1]
            ]
            if not prev_block.committed:
                raise ValueError("previous committed block is not in prefix cache")
            prefix_hash = prev_block.hash

        if computed_start < computed_end:
            first_block = computed_start // self.block_size
            last_block = (computed_end - 1) // self.block_size
            for block_idx in range(first_block, last_block + 1):
                block_start = block_idx * self.block_size
                block_id = block_table[block_idx]
                self._mark_block_computed(
                    block_id,
                    block_start,
                    computed_start,
                    computed_end,
                )

        new_committed_until = committed_until
        current_hash = prefix_hash
        while new_committed_until + self.block_size <= computed_end:
            block_idx = new_committed_until // self.block_size
            block_id = block_table[block_idx]
            block = self.blocks[block_id]
            if block.computed_tokens != self.block_size:
                break

            block_start = new_committed_until
            block_end = block_start + self.block_size
            block_tokens = token_ids[block_start:block_end]
            current_hash = self._commit_block(
                block_id,
                block_tokens,
                current_hash,
            )
            new_committed_until = block_end

        return new_committed_until, current_hash

    def commit_computed_prefill_blocks(
        self,
        block_table: List[int],
        token_ids: List[int],
        start: int,
        end: int,
        committed_until: int,
        prefix_hash: int = -1,
    ) -> tuple[int, int]:
        """Compatibility wrapper for chunked prefill commit."""
        return self.commit_computed_blocks(
            block_table=block_table,
            token_ids=token_ids,
            computed_start=start,
            computed_end=end,
            committed_until=committed_until,
            prefix_hash=prefix_hash,
        )

    def match_prefix_blocks(
        self,
        token_ids: List[int],
    ) -> tuple[List[int], int, int]:
        """Return committed prefix-cache blocks matching token_ids."""
        block_table = []
        num_cached_tokens = 0
        prefix_hash = -1
        num_full_blocks = len(token_ids) // self.block_size

        for block_idx in range(num_full_blocks):
            start_idx = block_idx * self.block_size
            end_idx = start_idx + self.block_size
            block_tokens = token_ids[start_idx:end_idx]
            current_hash = self.compute_hash(block_tokens, prefix_hash)
            cached_block_id = self.hash_to_block_id.get(current_hash, -1)
            if cached_block_id == -1:
                break

            block = self.blocks[cached_block_id]
            if not block.committed or block.token_ids != block_tokens:
                break

            block.ref_count += 1
            block_table.append(cached_block_id)
            num_cached_tokens += self.block_size
            prefix_hash = current_hash

        return block_table, num_cached_tokens, prefix_hash

    def _build_slot_mapping(
        self,
        block_table: List[int],
        start: int,
        end: int,
    ) -> List[int]:
        slot_mapping = []
        for token_pos in range(start, end):
            block_index = token_pos // self.block_size
            if block_index >= len(block_table):
                raise RuntimeError("block_table does not cover requested token range")
            block_offset = token_pos % self.block_size
            block_id = block_table[block_index]
            slot_mapping.append(block_id * self.block_size + block_offset)
        return slot_mapping

    def allocate_blocks(
        self, token_ids: List[int], block_table: List[int] = None
    ) -> tuple[List[int], List[int], int]:
        """Allocate cache blocks for new request with prefix caching support.

        Args:
            token_ids: Input token sequence
            block_table: Existing block_table (for decode phase)

        Returns:
            Tuple of (block_table, slot_mapping, num_cached_tokens)
        """
        if block_table is None:
            block_table = []

        num_tokens = len(token_ids)
        num_blocks = (num_tokens + self.block_size - 1) // self.block_size
        slot_mapping = []
        num_cached_tokens = 0
        prefix_hash = -1
        cache_miss = False

        for block_idx in range(num_blocks):
            start_idx = block_idx * self.block_size
            end_idx = min(start_idx + self.block_size, num_tokens)
            block_tokens = token_ids[start_idx:end_idx]

            # Only full blocks can be hashed for reuse
            if len(block_tokens) == self.block_size:
                prefix_hash = self.compute_hash(block_tokens, prefix_hash)

                # Try to reuse existing block
                if not cache_miss:
                    cached_block_id = self.hash_to_block_id.get(prefix_hash, -1)
                    if (
                        cached_block_id != -1
                        and self.blocks[cached_block_id].committed
                        and self.blocks[cached_block_id].token_ids == block_tokens
                    ):
                        # Check if all tokens are cached
                        if num_cached_tokens + self.block_size == len(token_ids):
                            cache_miss = True
                        else:
                            # Reuse successful
                            block = self.blocks[cached_block_id]
                            block.ref_count += 1
                            block_table.append(cached_block_id)
                            num_cached_tokens += self.block_size
                            continue
                    else:
                        cache_miss = True
            else:
                prefix_hash = -1

            # Cannot reuse, allocate new block
            if not self.free_block_ids:
                raise RuntimeError("No available cache blocks")

            new_block_id = self.free_block_ids[0]
            if prefix_hash != -1:
                block = self._allocate_full_block(new_block_id)
                block.update(prefix_hash, block_tokens)
            else:
                block = self._allocate_partial_block(new_block_id)
            block_table.append(new_block_id)

            # Generate slot_mapping
            for i in range(len(block_tokens)):
                slot_mapping.append(new_block_id * self.block_size + i)

        return block_table, slot_mapping, num_cached_tokens

    def append_slot(
        self, block_table: List[int], num_tokens: int, total_token_ids: List[int] = None
    ) -> tuple[List[int], int]:
        """Append slot for decode phase (generate one new token).

        Args:
            block_table: Current block_table
            num_tokens: Current total token count (including newly generated token)
            total_token_ids: All token sequence (for updating block hash)

        Returns:
            Tuple of (block_table, slot_id)
        """
        assert len(block_table) > 0, "block_table cannot be empty"
        assert num_tokens > 0, "num_tokens must be greater than 0"

        if num_tokens % self.block_size == 1:
            # Previous block is full, update its hash for future prefix caching
            last_block_id = block_table[-1]
            last_block = self.blocks[last_block_id]

            # Only update if block's token_ids is empty (avoid duplicate updates)
            if len(last_block.token_ids) == 0:
                block_start_idx = num_tokens - self.block_size - 1
                block_end_idx = num_tokens - 1
                block_tokens = total_token_ids[block_start_idx:block_end_idx]

                # Compute prefix_hash using previous block's hash if available
                if len(block_table) > 1:
                    prev_block = self.blocks[block_table[-2]]
                    prefix_hash = prev_block.hash
                else:
                    prefix_hash = -1

                current_hash = self.compute_hash(block_tokens, prefix_hash)
                last_block.update(current_hash, block_tokens, committed=True)
                self.hash_to_block_id[current_hash] = last_block_id

            # Need new block
            if not self.free_block_ids:
                if not self.try_free_blocks(1):
                    raise RuntimeError("No available cache blocks")
            new_block_id = self.free_block_ids[0]
            self._allocate_partial_block(new_block_id)
            block_table.append(new_block_id)

        # Calculate slot
        last_block_id = block_table[-1]
        offset = (num_tokens - 1) % self.block_size
        slot_id = last_block_id * self.block_size + offset

        return block_table, slot_id

    def free_blocks(self, block_table: List[int]):
        """Decrease reference count for all blocks. Blocks with ref_count=0 are not
        immediately freed to allow reuse."""
        for block_id in reversed(block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1

    def try_free_blocks(self, num_required: int) -> bool:
        """Try to free blocks with ref_count=0."""
        to_free = [
            bid for bid in self.used_block_ids if self.blocks[bid].ref_count == 0
        ]

        for block_id in to_free:
            self._deallocate_block(block_id)
            if self.can_allocate(num_required):
                return True

        return self.can_allocate(num_required)

    def get_num_free_blocks(self) -> int:
        return len(self.free_block_ids)

    def get_total_usable_blocks(self) -> int:
        freeable_used_blocks = sum(
            1 for bid in self.used_block_ids if self.blocks[bid].ref_count == 0
        )
        return len(self.free_block_ids) + freeable_used_blocks

    def __repr__(self):
        return (
            f"BlockManager(blocks={self.num_blocks}, block_size={self.block_size}, "
            f"free={len(self.free_block_ids)}, used={len(self.used_block_ids)})"
        )
