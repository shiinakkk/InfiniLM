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

PHASE_DECODE = "decode"
PHASE_PREFILL_MIDDLE = "prefill_middle"
PHASE_PREFILL_LAST = "prefill_last"
PREFILL_PHASES = {PHASE_PREFILL_MIDDLE, PHASE_PREFILL_LAST}
DEFAULT_MAX_NUM_BATCHED_TOKENS = 2048


class SchedulerOutput:
    """Scheduler output containing scheduled requests and execution phase info."""

    def __init__(
        self,
        scheduled_requests: List[InferenceRequest],
        is_prefill: bool = False,
        sample_output: bool = True,
        use_chunked_prefill: bool = False,
        request_phases: Optional[List[str]] = None,
        sample_mask: Optional[List[bool]] = None,
        input_ranges: Optional[List[Optional[tuple[int, int]]]] = None,
    ):
        self.scheduled_requests = scheduled_requests
        self.num_requests = len(scheduled_requests)
        self.is_prefill = is_prefill
        self.sample_output = sample_output
        self.use_chunked_prefill = use_chunked_prefill
        if request_phases is None:
            phase = (
                PHASE_PREFILL_LAST
                if is_prefill and sample_output
                else PHASE_PREFILL_MIDDLE
                if is_prefill
                else PHASE_DECODE
            )
            request_phases = [phase] * self.num_requests
        if sample_mask is None:
            sample_mask = [
                phase in (PHASE_DECODE, PHASE_PREFILL_LAST)
                for phase in request_phases
            ]
        if input_ranges is None:
            input_ranges = [None] * self.num_requests

        if len(request_phases) != self.num_requests:
            raise ValueError("request_phases length must match scheduled_requests")
        if len(sample_mask) != self.num_requests:
            raise ValueError("sample_mask length must match scheduled_requests")
        if len(input_ranges) != self.num_requests:
            raise ValueError("input_ranges length must match scheduled_requests")

        self.request_phases = request_phases
        self.sample_mask = sample_mask
        self.input_ranges = input_ranges


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
        prefill_chunk_size: int = 2048,
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
            self.max_num_batched_tokens = DEFAULT_MAX_NUM_BATCHED_TOKENS
        self._prefill_rr_next_source = "running"

        self.cache_manager = BlockManager(num_blocks=num_blocks, block_size=block_size)
        self.block_size = block_size

    def add_request(self, request: InferenceRequest):
        if request is not None:
            request.status = RequestStatus.WAITING
            self.waiting_queue.sync_q.put(request)

    def schedule(self) -> Optional[SchedulerOutput]:
        """Schedule and return batch of requests to execute."""
        if self.enable_continuous_batching:
            return self._schedule_continuous()

        scheduled_requests = []
        is_prefill = False
        sample_output = True
        request_phases = []
        sample_mask = []
        input_ranges = []

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
                phase = (
                    PHASE_PREFILL_LAST
                    if req.should_sample_current_step()
                    else PHASE_PREFILL_MIDDLE
                )
                request_phases.append(phase)
                sample_mask.append(req.should_sample_current_step())
                input_ranges.append((req.prefill_chunk_start, req.prefill_chunk_end))
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
                request_phases.append(PHASE_PREFILL_LAST)
                sample_mask.append(True)
                input_ranges.append((req.num_cached_tokens, len(req_tokens)))

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
                request_phases=request_phases,
                sample_mask=sample_mask,
                input_ranges=input_ranges,
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
                phase = (
                    PHASE_PREFILL_LAST
                    if req.should_sample_current_step()
                    else PHASE_PREFILL_MIDDLE
                )
                request_phases.append(phase)
                sample_mask.append(req.should_sample_current_step())
                input_ranges.append((req.prefill_chunk_start, req.prefill_chunk_end))

            if scheduled_requests:
                is_prefill = True
                return SchedulerOutput(
                    scheduled_requests=scheduled_requests,
                    is_prefill=is_prefill,
                    sample_output=sample_output,
                    use_chunked_prefill=True,
                    request_phases=request_phases,
                    sample_mask=sample_mask,
                    input_ranges=input_ranges,
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
                request_phases.append(PHASE_DECODE)
                sample_mask.append(True)
                input_ranges.append(None)

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
                request_phases=request_phases,
                sample_mask=sample_mask,
                input_ranges=input_ranges,
            )

        return None

    def _schedule_continuous(self) -> Optional[SchedulerOutput]:
        """Schedule a mixed decode/prefill batch for continuous batching."""
        scheduled_requests: List[InferenceRequest] = []
        request_phases: List[str] = []
        sample_mask: List[bool] = []
        input_ranges: List[Optional[tuple[int, int]]] = []
        used_tokens = 0

        used_tokens = self._schedule_decode_requests(
            scheduled_requests,
            request_phases,
            sample_mask,
            input_ranges,
            used_tokens,
        )
        self._schedule_prefill_round_robin_requests(
            scheduled_requests,
            request_phases,
            sample_mask,
            input_ranges,
            used_tokens,
        )

        if not scheduled_requests:
            return None

        return SchedulerOutput(
            scheduled_requests=scheduled_requests,
            is_prefill=all(phase in PREFILL_PHASES for phase in request_phases),
            sample_output=any(sample_mask),
            use_chunked_prefill=True,
            request_phases=request_phases,
            sample_mask=sample_mask,
            input_ranges=input_ranges,
        )

    def _schedule_decode_requests(
        self,
        scheduled_requests: List[InferenceRequest],
        request_phases: List[str],
        sample_mask: List[bool],
        input_ranges: List[Optional[tuple[int, int]]],
        used_tokens: int,
    ) -> int:
        running_queue_size = self.running_queue.sync_q.qsize()
        for _ in range(running_queue_size):
            if not self._fits_token_budget(1, used_tokens, bool(scheduled_requests)):
                break

            try:
                req = self.running_queue.sync_q.get_nowait()
            except queue.Empty:
                break

            if req.is_finished():
                self.complete_requests([req])
                continue

            if req.is_prefill:
                self.running_queue.sync_q.put(req)
                continue

            try:
                req.block_table, new_slot = self.cache_manager.append_slot(
                    req.block_table, req.get_total_length(), req.get_all_token_ids()
                )
            except RuntimeError as e:
                raise RuntimeError("No available cache blocks for new token") from e

            req.slot_mapping = [new_slot]
            req.num_blocks = len(req.block_table)
            req.num_cached_tokens = req.get_total_length() - 1
            req.status = RequestStatus.RUNNING

            scheduled_requests.append(req)
            request_phases.append(PHASE_DECODE)
            sample_mask.append(True)
            input_ranges.append(None)
            used_tokens += 1

        return used_tokens

    def _schedule_prefill_round_robin_requests(
        self,
        scheduled_requests: List[InferenceRequest],
        request_phases: List[str],
        sample_mask: List[bool],
        input_ranges: List[Optional[tuple[int, int]]],
        used_tokens: int,
    ) -> int:
        """Schedule prefill chunks with round-robin fairness across requests."""
        failed_sources = set()

        while True:
            remaining_budget = self._remaining_token_budget(used_tokens)
            if remaining_budget is not None and remaining_budget <= 0:
                break

            source = self._prefill_rr_next_source
            if source == "running":
                chunk_len = self._schedule_one_running_prefill_request(
                    scheduled_requests,
                    request_phases,
                    sample_mask,
                    input_ranges,
                    used_tokens,
                )
                self._prefill_rr_next_source = "waiting"
            else:
                chunk_len = self._schedule_one_waiting_prefill_request(
                    scheduled_requests,
                    request_phases,
                    sample_mask,
                    input_ranges,
                    used_tokens,
                )
                self._prefill_rr_next_source = "running"

            if chunk_len is None:
                failed_sources.add(source)
                if len(failed_sources) >= 2:
                    break
                continue

            failed_sources.clear()
            used_tokens += chunk_len

        return used_tokens

    def _schedule_one_running_prefill_request(
        self,
        scheduled_requests: List[InferenceRequest],
        request_phases: List[str],
        sample_mask: List[bool],
        input_ranges: List[Optional[tuple[int, int]]],
        used_tokens: int,
    ) -> Optional[int]:
        running_queue_size = self.running_queue.sync_q.qsize()
        for _ in range(running_queue_size):
            remaining_budget = self._remaining_token_budget(used_tokens)
            if remaining_budget is not None and remaining_budget <= 0:
                return None

            try:
                req = self.running_queue.sync_q.get_nowait()
            except queue.Empty:
                return None

            if req.is_finished():
                self.complete_requests([req])
                continue

            if not req.is_prefill:
                self.running_queue.sync_q.put(req)
                continue

            chunk_len = self._prepare_chunked_prefill_request(
                req, max_chunk_tokens=remaining_budget
            )
            if chunk_len <= 0:
                self.running_queue.sync_q.put(req)
                continue

            self._append_prefill_schedule(
                req,
                scheduled_requests,
                request_phases,
                sample_mask,
                input_ranges,
            )
            return chunk_len

        return None

    def _schedule_one_waiting_prefill_request(
        self,
        scheduled_requests: List[InferenceRequest],
        request_phases: List[str],
        sample_mask: List[bool],
        input_ranges: List[Optional[tuple[int, int]]],
        used_tokens: int,
    ) -> Optional[int]:
        while True:
            remaining_budget = self._remaining_token_budget(used_tokens)
            if remaining_budget is not None and remaining_budget <= 0:
                return None

            try:
                req = self.waiting_queue.sync_q.get_nowait()
            except queue.Empty:
                return None

            if req.is_finished():
                self.complete_requests([req])
                continue

            if not self.can_accept_request(
                req, additional_requests=scheduled_requests
            ):
                self.waiting_queue.sync_q.put(req)
                return None

            self._initialize_chunked_prefill_cache(req)
            chunk_len = self._prepare_chunked_prefill_request(
                req, max_chunk_tokens=remaining_budget
            )
            if chunk_len <= 0:
                self.waiting_queue.sync_q.put(req)
                return None

            self._append_prefill_schedule(
                req,
                scheduled_requests,
                request_phases,
                sample_mask,
                input_ranges,
            )
            return chunk_len

        return None

    def _append_prefill_schedule(
        self,
        req: InferenceRequest,
        scheduled_requests: List[InferenceRequest],
        request_phases: List[str],
        sample_mask: List[bool],
        input_ranges: List[Optional[tuple[int, int]]],
    ) -> None:
        req.status = RequestStatus.RUNNING
        scheduled_requests.append(req)
        phase = (
            PHASE_PREFILL_LAST
            if req.should_sample_current_step()
            else PHASE_PREFILL_MIDDLE
        )
        request_phases.append(phase)
        sample_mask.append(req.should_sample_current_step())
        input_ranges.append((req.prefill_chunk_start, req.prefill_chunk_end))

    def _batch_is_full(self, scheduled_requests: List[InferenceRequest]) -> bool:
        return len(scheduled_requests) >= self.max_batch_size

    def _remaining_token_budget(self, used_tokens: int) -> Optional[int]:
        if self.max_num_batched_tokens is None:
            return None
        return self.max_num_batched_tokens - used_tokens

    def _fits_token_budget(
        self,
        num_tokens: int,
        used_tokens: int,
        has_scheduled_requests: bool,
    ) -> bool:
        if self.max_num_batched_tokens is None:
            return True
        if num_tokens <= self.max_num_batched_tokens - used_tokens:
            return True
        return not has_scheduled_requests and num_tokens <= self.max_num_batched_tokens

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

    def _chunked_prefill_needs_sampling(
        self,
        req: InferenceRequest,
        max_chunk_tokens: Optional[int] = None,
    ) -> bool:
        if self.prefill_chunk_size <= 0:
            raise ValueError("prefill_chunk_size must be greater than 0")

        chunk_size = self.prefill_chunk_size
        if max_chunk_tokens is not None:
            chunk_size = min(chunk_size, max_chunk_tokens)
        if chunk_size <= 0:
            return False

        end = min(
            req.num_prompt_tokens_computed + chunk_size,
            req.prompt_length,
        )
        return end == req.prompt_length

    def _prepare_chunked_prefill_request(
        self,
        req: InferenceRequest,
        max_chunk_tokens: Optional[int] = None,
    ) -> int:
        if self.prefill_chunk_size <= 0:
            raise ValueError("prefill_chunk_size must be greater than 0")

        chunk_size = self.prefill_chunk_size
        if max_chunk_tokens is not None:
            chunk_size = min(chunk_size, max_chunk_tokens)
        if chunk_size <= 0:
            return 0

        start = req.num_prompt_tokens_computed
        end = min(start + chunk_size, req.prompt_length)
        if end <= start:
            return 0

        req.mark_prefill_chunk(start, end)
        logger.debug(
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
        return end - start

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

    def can_accept_request(
        self,
        request: InferenceRequest,
        additional_requests: Optional[List[InferenceRequest]] = None,
    ) -> bool:
        total_required_blocks = 0

        additional_requests = additional_requests or []
        for req in additional_requests:
            if req.is_finished():
                continue
            remaining_tokens = (
                req.sampling_params.max_tokens - req.get_num_generated_tokens()
            )
            num_blocks_needed = (
                remaining_tokens + self.block_size - 1
            ) // self.block_size
            total_required_blocks += num_blocks_needed

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
