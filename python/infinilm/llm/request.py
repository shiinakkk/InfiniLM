"""
Request and Output - Data structures for inference requests and outputs.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Any
import time
import janus
import asyncio
import logging

from infinilm.llm.sampling_params import SamplingParams

logger = logging.getLogger(__name__)


class RequestStatus(Enum):
    """Status of an inference request."""

    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    CANCELED = "canceled"
    FAILED = "failed"
    TIMEOUT = "timeout"


class FinishReason(Enum):
    """Reason for finishing generation."""

    STOP = "stop"
    LENGTH = "length"
    EOS_TOKEN = "eos_token"
    STOP_STRING = "stop_string"
    TIMEOUT = "timeout"
    CANCELED = "canceled"
    ERROR = "error"


@dataclass
class RequestOutput:
    """Output from a single generation request.

    Attributes:
        request_id: Unique identifier for the request.
        prompt: Original prompt text.
        prompt_token_ids: Token IDs of the prompt.
        outputs: List of generated outputs (for beam search, multiple outputs possible).
        finished: Whether generation is complete.
        finish_reason: Reason for finishing.
    """

    request_id: str
    prompt: Optional[str] = None
    prompt_token_ids: Optional[List[int]] = None
    outputs: List["CompletionOutput"] = field(default_factory=list)
    finished: bool = False
    finish_reason: Optional[FinishReason] = None


@dataclass
class CompletionOutput:
    """Single completion output.

    Attributes:
        index: Index of this output (for beam search).
        text: Generated text.
        token_ids: Generated token IDs.
        finish_reason: Reason for finishing.
    """

    index: int = 0
    text: str = ""
    token_ids: List[int] = field(default_factory=list)
    finish_reason: Optional[FinishReason] = None


@dataclass
class TokenOutput:
    """Output for a single generated token.

    Attributes:
        request_id: Unique identifier for the request.
        token_id: Generated token ID.
        token_text: Decoded text of the token.
        finished: Whether generation is complete.
        finish_reason: Reason for finishing.
        generated_text: Full generated text so far.
    """

    request_id: str
    token_id: int
    token_text: str
    finished: bool = False
    finish_reason: Optional[FinishReason] = None
    generated_text: str = ""


class InferenceRequest:
    """Internal inference request object for managing generation state and resources."""

    def __init__(
        self,
        request_id: str,
        prompt: Optional[str] = None,
        prompt_token_ids: Optional[List[int]] = None,
        processed_inputs: Optional[dict] = None,
        sampling_params: Optional[SamplingParams] = None,
        eos_token_ids: Optional[List[int]] = None,
        arrival_time: Optional[float] = None,
        # For server use
        request_data: Optional[dict] = None,
        http_request: Optional[Any] = None,
    ):
        self.arrival_time: float = arrival_time or time.time()
        self.finished_time: Optional[float] = None

        # Request metadata
        self.request_id: str = request_id
        self.prompt: Optional[str] = prompt
        self.prompt_token_ids: List[int] = prompt_token_ids or []
        self.prompt_length: int = len(self.prompt_token_ids)
        self.processed_inputs: Optional[dict] = processed_inputs

        # Sampling parameters
        self.sampling_params: SamplingParams = sampling_params or SamplingParams()

        # EOS token IDs (from model config)
        self.eos_token_ids: List[int] = eos_token_ids or []

        # Generation state
        self.generated_token_ids: List[int] = []
        self.generated_text: str = ""
        self.is_prefill: bool = True
        self.status: RequestStatus = RequestStatus.WAITING
        self.num_prompt_tokens_computed: int = 0
        self.prefill_chunk_start: int = 0
        self.prefill_chunk_end: int = 0
        self.prefill_done: bool = self.prompt_length == 0
        self.needs_sampling: bool = True
        self.num_prompt_tokens_committed: int = 0
        self.last_committed_block_hash: int = -1
        self.finish_reason: Optional[FinishReason] = None
        self.priority: int = 0

        # KV cache management
        self.cache_id: Optional[int] = None
        self.block_table: List[int] = []
        self.slot_mapping: List[int] = []
        self.num_cached_tokens: int = 0
        self.num_blocks: int = 0

        # For server use
        self.request_data: Optional[dict] = request_data
        self.http_request: Optional[Any] = http_request

        # Output management (for async streaming)
        self._output_queue: Optional[janus.Queue] = None
        self._aborted = False

        # Streaming helpers (vLLM-style UTF-8 buffering at the chunking layer)
        # Used by the engine to compute "delta" text chunks from a full decode.
        self._stream_last_yielded_length: int = 0
        self._pending_token_offset: int = 0

    @property
    def output_queue(self) -> janus.Queue:
        """Lazy initialization of output queue."""
        if self._output_queue is None:
            self._output_queue = janus.Queue()
        return self._output_queue

    def get_prompt_length(self) -> int:
        return self.prompt_length

    def get_input_tokens(self) -> List[int]:
        return self.prompt_token_ids

    def get_num_generated_tokens(self) -> int:
        return len(self.generated_token_ids)

    def get_num_prompt_tokens_remaining(self) -> int:
        """Return prompt tokens that still need prefill computation."""
        return max(self.prompt_length - self.num_prompt_tokens_computed, 0)

    def get_num_prompt_tokens_uncommitted(self) -> int:
        """Return computed prompt tokens that have not entered prefix cache."""
        return max(
            self.num_prompt_tokens_computed - self.num_prompt_tokens_committed,
            0,
        )

    def get_prefill_chunk_tokens(self) -> List[int]:
        """Return token IDs for the currently scheduled prefill chunk."""
        return self.prompt_token_ids[self.prefill_chunk_start : self.prefill_chunk_end]

    def mark_prefill_chunk(
        self,
        start: int,
        end: int,
        needs_sampling: Optional[bool] = None,
    ) -> None:
        """Record the prompt token range scheduled for the next prefill step."""
        if start < 0:
            raise ValueError("prefill chunk start must be non-negative")
        if end < start:
            raise ValueError("prefill chunk end must be greater than or equal to start")
        if end > self.prompt_length:
            raise ValueError("prefill chunk end cannot exceed prompt length")
        if start < self.num_prompt_tokens_computed:
            raise ValueError(
                "prefill chunk start cannot be before computed prompt progress"
            )

        self.prefill_chunk_start = start
        self.prefill_chunk_end = end
        self.needs_sampling = (
            end == self.prompt_length if needs_sampling is None else needs_sampling
        )

    def mark_prefill_progress(self, end: int) -> None:
        """Advance prompt prefill progress after a chunk has been computed."""
        if end < self.num_prompt_tokens_computed:
            raise ValueError("prefill progress cannot move backwards")
        if end > self.prompt_length:
            raise ValueError("prefill progress cannot exceed prompt length")

        self.num_prompt_tokens_computed = end
        self.prefill_done = end == self.prompt_length

    def mark_prefill_committed(
        self,
        end: int,
        prefix_hash: Optional[int] = None,
        block_size: Optional[int] = None,
    ) -> None:
        """Advance prefix-cache commit progress for computed prompt tokens."""
        if end < self.num_prompt_tokens_committed:
            raise ValueError("prefill commit progress cannot move backwards")
        if end > self.num_prompt_tokens_computed:
            raise ValueError("prefill commit progress cannot exceed computed progress")
        if end > self.prompt_length:
            raise ValueError("prefill commit progress cannot exceed prompt length")
        if block_size is not None and end % block_size != 0:
            raise ValueError("prefill commit progress must stop on a block boundary")

        self.num_prompt_tokens_committed = end
        if prefix_hash is not None:
            self.last_committed_block_hash = prefix_hash

    def validate_prefill_cache_progress(
        self, block_size: Optional[int] = None
    ) -> None:
        """Validate computed and committed prompt cache progress invariants."""
        if not (
            0
            <= self.num_prompt_tokens_committed
            <= self.num_prompt_tokens_computed
            <= self.prompt_length
        ):
            raise ValueError("invalid prefill cache progress ordering")
        if (
            block_size is not None
            and self.num_prompt_tokens_committed % block_size != 0
        ):
            raise ValueError("committed prompt progress must align to block boundary")

    def is_prefill_complete(self) -> bool:
        return self.prefill_done

    def should_sample_current_step(self) -> bool:
        return self.needs_sampling

    def get_total_length(self) -> int:
        return self.prompt_length + len(self.generated_token_ids)

    def get_all_token_ids(self) -> List[int]:
        return self.prompt_token_ids + self.generated_token_ids

    def get_num_blocks_required(self, block_size: int) -> int:
        total_tokens = self.get_total_length()
        return (total_tokens + block_size - 1) // block_size

    def get_max_tokens(self) -> Optional[int]:
        return self.sampling_params.max_tokens

    def is_finished(self) -> bool:
        return self.status in [
            RequestStatus.FINISHED,
            RequestStatus.CANCELED,
            RequestStatus.FAILED,
            RequestStatus.TIMEOUT,
        ]

    def abort(self):
        """Signal that the request has been aborted and should stop generation."""
        self._aborted = True

    def is_aborted(self) -> bool:
        """Check if the request has been aborted."""
        return self._aborted

    def mark_finished(self, reason: FinishReason):
        """Mark the request as finished with the given reason."""
        self.status = RequestStatus.FINISHED
        self.finish_reason = reason
        self.finished_time = time.time()

    def mark_failed(self, reason: FinishReason = FinishReason.ERROR):
        """Mark the request as failed."""
        self.abort()
        self.status = RequestStatus.FAILED
        self.finish_reason = reason
        self.finished_time = time.time()

    def mark_canceled(self):
        """Mark the request as canceled."""
        self.abort()
        self.status = RequestStatus.CANCELED
        self.finish_reason = FinishReason.CANCELED
        self.finished_time = time.time()

    def mark_timeout(self):
        """Mark the request as timed out."""
        self.abort()
        self.status = RequestStatus.TIMEOUT
        self.finish_reason = FinishReason.TIMEOUT
        self.finished_time = time.time()

    async def close(self):
        """Close the output queue and clean up resources."""
        if self._output_queue is not None:
            self.abort()
            try:
                while not self._output_queue.async_q.empty():
                    try:
                        self._output_queue.async_q.get_nowait()
                        self._output_queue.async_q.task_done()
                    except asyncio.QueueEmpty:
                        break
            except Exception as e:
                logger.error(
                    f"Error while clearing output queue for request {self.request_id}: {e}"
                )
                pass

            self._output_queue.close()
            try:
                await asyncio.wait_for(self._output_queue.wait_closed(), timeout=0.5)
            except asyncio.TimeoutError:
                logger.warning("wait_closed timeout, force close")

    def to_request_output(self) -> RequestOutput:
        """Convert to RequestOutput for external use."""
        return RequestOutput(
            request_id=self.request_id,
            prompt=self.prompt,
            prompt_token_ids=self.prompt_token_ids,
            outputs=[
                CompletionOutput(
                    index=0,
                    text=self.generated_text,
                    token_ids=self.generated_token_ids.copy(),
                    finish_reason=self.finish_reason,
                )
            ],
            finished=self.is_finished(),
            finish_reason=self.finish_reason,
        )
