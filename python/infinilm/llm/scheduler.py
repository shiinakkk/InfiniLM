"""
Scheduler - Request scheduling and batch management with Paged Attention KV Cache.
"""

import queue
import janus
import logging
from typing import List, Optional
from infinilm.llm.request import RequestStatus, InferenceRequest
from infinilm.llm.cache_manager import BlockManager

logger = logging.getLogger(__name__)


class SchedulerOutput:
    """Scheduler output containing scheduled requests and execution phase info."""

    def __init__(
        self,
        scheduled_requests: List[InferenceRequest],
        is_prefill: bool = False,
        sample_output: bool = True,
        use_chunked_prefill: bool = False,
    ):
        self.scheduled_requests = scheduled_requests
        self.num_requests = len(scheduled_requests)
        self.is_prefill = is_prefill
        self.sample_output = sample_output
        self.use_chunked_prefill = use_chunked_prefill

class Scheduler:
    """Request scheduler with integrated BlockManager for KV cache management.

    Scheduling logic:
    1. Running queue: Check for new blocks needed, update slot_mapping
    2. Waiting queue: Try block reuse (prefix caching), allocate new blocks
    3. Reference counting: Free blocks when requests complete
    """

    def __init__(
        self,
        max_batch_size: int = 16,
        num_blocks: int = 512,
        block_size: int = 256,
        enable_chunked_prefill: bool = False,
        prefill_chunk_size: int = 512,
        enable_continuous_batching: bool = False,
        max_num_batched_tokens: Optional[int] = None,
    ):
        if max_num_batched_tokens is not None and max_num_batched_tokens <= 0:
            raise ValueError("max_num_batched_tokens must be greater than 0")
        if enable_continuous_batching and not enable_chunked_prefill:
            raise ValueError(
                "enable_continuous_batching currently requires enable_chunked_prefill"
            )

        self.waiting_queue = janus.Queue()
        self.running_queue = janus.Queue()
        self.max_batch_size = max_batch_size
        self.enable_chunked_prefill = enable_chunked_prefill
        self.prefill_chunk_size = prefill_chunk_size
        self.enable_continuous_batching = enable_continuous_batching
        self.max_num_batched_tokens = max_num_batched_tokens
        if self.enable_continuous_batching and self.max_num_batched_tokens is None:
            self.max_num_batched_tokens = self.max_batch_size * self.prefill_chunk_size

        self.cache_manager = BlockManager(num_blocks=num_blocks, block_size=block_size)
        self.block_size = block_size

    def add_request(self, request: InferenceRequest):
        if request is not None:
            request.status = RequestStatus.WAITING
            self.waiting_queue.sync_q.put(request)

    def schedule(self) -> Optional[SchedulerOutput]:
        """Schedule and return batch of requests to execute."""
        scheduled_requests = []
        is_prefill = False
        sample_output = True

        # Process Waiting queue (prefill phase)
        while len(scheduled_requests) < self.max_batch_size:
            try:
                req = self.waiting_queue.sync_q.get_nowait()
            except queue.Empty:
                break
            # Skip requests that were already finished (e.g., timed out/canceled while waiting)
            if req.is_finished():
                self.complete_requests([req])
                continue

            if not self.can_accept_request(req):
                self.waiting_queue.sync_q.put(req)
                break

            # Skip requests that were already finished (e.g., timed out/canceled while waiting)
            if req.is_finished():
                self.complete_requests([req])
                continue

            if self.enable_chunked_prefill:
                self._initialize_chunked_prefill_cache(req)
                req_sample_output = self._chunked_prefill_needs_sampling(req)
                if scheduled_requests and req_sample_output != sample_output:
                    self.waiting_queue.sync_q.put(req)
                    break
                sample_output = req_sample_output
                self._prepare_chunked_prefill_request(req)
            else:
                req_tokens = req.get_input_tokens()
                num_required_blocks = req.get_num_blocks_required(self.block_size)

                if not self.cache_manager.can_allocate(num_required_blocks):
                    if not self.cache_manager.try_free_blocks(num_required_blocks):
                        raise RuntimeError("No available cache blocks for new request")

                # Allocate blocks with automatic prefix caching support
                req.block_table, req.slot_mapping, req.num_cached_tokens = (
                    self.cache_manager.allocate_blocks(req_tokens, req.block_table)
                )

            req.num_blocks = len(req.block_table)
            req.status = RequestStatus.RUNNING
            scheduled_requests.append(req)

        # Return prefill batch if any waiting requests were scheduled
        if scheduled_requests:
            is_prefill = True
            return SchedulerOutput(
                scheduled_requests=scheduled_requests,
                is_prefill=is_prefill,
                sample_output=sample_output,
                use_chunked_prefill=self.enable_chunked_prefill,
            )

        # Process prefill continuations before decode. Step 1 keeps prefill and decode
        # in separate forward batches, so a request still in prefill remains prefill.
        if self.enable_chunked_prefill:
            running_queue_size = self.running_queue.sync_q.qsize()
            for _ in range(running_queue_size):
                if len(scheduled_requests) >= self.max_batch_size:
                    break
                try:
                    req = self.running_queue.sync_q.get_nowait()
                except queue.Empty:
                    break

                if req.is_finished():
                    self.complete_requests([req])
                    continue

                if not req.is_prefill:
                    self.running_queue.sync_q.put(req)
                    continue

                req_sample_output = self._chunked_prefill_needs_sampling(req)
                if scheduled_requests and req_sample_output != sample_output:
                    self.running_queue.sync_q.put(req)
                    break
                sample_output = req_sample_output
                self._prepare_chunked_prefill_request(req)
                req.status = RequestStatus.RUNNING
                scheduled_requests.append(req)

            if scheduled_requests:
                is_prefill = True
                return SchedulerOutput(
                    scheduled_requests=scheduled_requests,
                    is_prefill=is_prefill,
                    sample_output=sample_output,
                    use_chunked_prefill=True,
                )

        # Process Running queue (decode phase)
        while len(scheduled_requests) < self.max_batch_size:
            try:
                req = self.running_queue.sync_q.get_nowait()
            except queue.Empty:
                break
            # Skip requests that were already finished (e.g., timed out/canceled while running)
            if req.is_finished():
                self.complete_requests([req])
                continue

            # Decode phase: allocate slot for newly generated token
            try:
                req.block_table, new_slot = self.cache_manager.append_slot(
                    req.block_table, req.get_total_length(), req.get_all_token_ids()
                )
                req.slot_mapping = [new_slot]
                req.num_blocks = len(req.block_table)
                req.num_cached_tokens = req.get_total_length() - 1
                scheduled_requests.append(req)

            except RuntimeError as e:
                raise RuntimeError("No available cache blocks for new token") from e

        # Return decode batch if any running requests were scheduled
        if scheduled_requests:
            is_prefill = False
            return SchedulerOutput(
                scheduled_requests=scheduled_requests,
                is_prefill=is_prefill,
                sample_output=True,
                use_chunked_prefill=False,
            )

        return None

    def _initialize_chunked_prefill_cache(self, req: InferenceRequest) -> None:
        """Attach committed prefix-cache blocks before the first chunk."""
        if req.block_table or req.num_prompt_tokens_computed:
            return

        # Keep at least the final prompt token in the forward path so the model
        # can produce logits for the first generated token.
        match_tokens = req.get_input_tokens()[:-1]
        block_table, num_cached_tokens, prefix_hash = (
            self.cache_manager.match_prefix_blocks(match_tokens)
        )
        if num_cached_tokens == 0:
            return

        req.block_table = block_table
        req.num_cached_tokens = num_cached_tokens
        req.num_blocks = len(block_table)
        req.mark_prefill_progress(num_cached_tokens)
        req.mark_prefill_committed(
            num_cached_tokens,
            prefix_hash,
            self.block_size,
        )
        req.validate_prefill_cache_progress(self.block_size)
        logger.debug(
            "chunked prefill prefix hit request=%s cached=%d blocks=%d",
            req.request_id[:8],
            num_cached_tokens,
            len(block_table),
        )

    def _chunked_prefill_needs_sampling(self, req: InferenceRequest) -> bool:
        if self.prefill_chunk_size <= 0:
            raise ValueError("prefill_chunk_size must be greater than 0")

        end = min(
            req.num_prompt_tokens_computed + self.prefill_chunk_size,
            req.prompt_length,
        )
        return end == req.prompt_length

    def _prepare_chunked_prefill_request(self, req: InferenceRequest) -> bool:
        if self.prefill_chunk_size <= 0:
            raise ValueError("prefill_chunk_size must be greater than 0")

        start = req.num_prompt_tokens_computed
        end = min(start + self.prefill_chunk_size, req.prompt_length)
        req.mark_prefill_chunk(start, end)
        logger.info(
            "chunked prefill request=%s chunk=[%d,%d) sample_output=%s",
            req.request_id[:8],
            start,
            end,
            req.should_sample_current_step(),
        )
        req.block_table, req.slot_mapping = self.cache_manager.allocate_prefill_chunk(
            req.block_table, start, end
        )
        req.num_cached_tokens = start
        req.num_blocks = len(req.block_table)
        return req.should_sample_current_step()

    def commit_prefill_progress(self, req: InferenceRequest) -> None:
        """Commit full prompt blocks after a chunked prefill forward succeeds."""
        if not self.enable_chunked_prefill:
            raise RuntimeError("commit_prefill_progress requires chunked prefill")

        committed_until, prefix_hash = (
            self.cache_manager.commit_computed_blocks(
                block_table=req.block_table,
                token_ids=req.get_input_tokens(),
                computed_start=req.prefill_chunk_start,
                computed_end=req.prefill_chunk_end,
                committed_until=req.num_prompt_tokens_committed,
                prefix_hash=req.last_committed_block_hash,
            )
        )
        req.mark_prefill_committed(
            committed_until,
            prefix_hash if committed_until > 0 else None,
            self.block_size,
        )
        req.validate_prefill_cache_progress(self.block_size)
        logger.debug(
            "chunked prefill cache request=%s computed=%d committed=%d hash=%s",
            req.request_id[:8],
            req.num_prompt_tokens_computed,
            req.num_prompt_tokens_committed,
            req.last_committed_block_hash,
        )

    def complete_requests(self, requests: List[InferenceRequest]):
        """Handle completed requests and free their blocks."""
        for req in requests:
            if req.status in [
                RequestStatus.FINISHED,
                RequestStatus.CANCELED,
                RequestStatus.FAILED,
                RequestStatus.TIMEOUT,
            ]:
                if req.block_table:
                    self.cache_manager.free_blocks(req.block_table)

                if req.status == RequestStatus.CANCELED:
                    logger.info(
                        f"Request {req.request_id[:8]}... canceled: {req.finish_reason}"
                    )
                elif req.status == RequestStatus.FAILED:
                    logger.error(
                        f"Request {req.request_id[:8]}... failed: {req.finish_reason}"
                    )
                elif req.status == RequestStatus.TIMEOUT:
                    logger.error(
                        f"Request {req.request_id[:8]}... timed out: {req.finish_reason}"
                    )
            else:
                # Still running, put back in running queue
                self.running_queue.sync_q.put(req)

    def can_accept_request(self, request: InferenceRequest) -> bool:
        total_required_blocks = 0

        # Calculate blocks needed for running requests
        running_queue_size = self.running_queue.sync_q.qsize()
        for _ in range(running_queue_size):
            req = self.running_queue.sync_q.get()
            remaining_tokens = (
                req.sampling_params.max_tokens - req.get_num_generated_tokens()
            )
            num_blocks_needed = (
                remaining_tokens + self.block_size - 1
            ) // self.block_size
            total_required_blocks += num_blocks_needed
            self.running_queue.sync_q.put(req)

        # Calculate blocks needed for the new request
        total_length = request.get_prompt_length()
        total_length += request.sampling_params.max_tokens
        num_blocks_needed = (total_length + self.block_size - 1) // self.block_size
        total_required_blocks += num_blocks_needed

        # Compare with total usable blocks in cache manager
        return total_required_blocks <= self.cache_manager.get_total_usable_blocks()

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "num_blocks": self.cache_manager.num_blocks,
            "block_size": self.cache_manager.block_size,
            "num_free_blocks": self.cache_manager.get_num_free_blocks(),
            "num_req_blocks": len(self.cache_manager.req_block_ids),
            "num_used_blocks": len(self.cache_manager.used_block_ids),
        }
