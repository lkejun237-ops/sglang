# SPDX-License-Identifier: Apache-2.0
"""Adapt queued DiT results to the existing remote Lynn decoder client."""

from .remote_decoder import create_decoder


class DiTDecoderOutput:
    def __init__(self, codecs, init, config, *, parallel=None):
        self.parallel = parallel
        self.provenance = None
        self.remote = create_decoder(
            codecs,
            policy={
                "remote": dict(
                    config,
                    auto_output=True,
                    output_session=dict(init),
                    oneway=init.get("async_inputs", False),
                )
            },
            capacity_latents=10,
            provenance_for=lambda pixels, dependency: self.provenance,
            session_id=init["session_id"],
        )
        if self.remote.failure is not None:
            self.remote.dispose()
            raise RuntimeError("Decoder connection failed") from self.remote.failure

    def output(self, result, message, item):
        self.provenance = result.provenance
        self.remote.output_metadata = dict(
            result.provenance.metadata(),
            event_id=result.prompt_event_id,
            prompt_version=result.prompt_version,
            sink_event_id=getattr(result, "sink_event_id", None),
            sink_chunks=getattr(result, "sink_chunks", 0),
            sink_chunk_ids=getattr(result, "sink_chunk_ids", ()),
            block_idx=result.chunk.chunk_id,
        )
        self.remote.bind_dispatch(
            request_id=message["request_id"],
            chunk_index=message["chunk_index"],
            generation_id=message["generation_id"],
        )
        self.remote.append(
            result.visual,
            latent_start=result.chunk.start,
            source_dependency_end=result.provenance.source_dependency_end,
            valid_frame_end=result.provenance.pixels.end,
        )
        decoded = self.remote.pop()
        return dict(
            {k: v for k, v in item.items() if k not in ("tensor", "format", "type")},
            # In one-way mode this describes transfer completion, never a
            # Decoder admission/model ACK. Decoder alone publishes RGB/credit.
            type="decoder.input_sent"
            if self.remote.oneway
            else "decoder.output_queued"
            if self.remote.output_attachment
            else "decoder.frames",
            format="rgb24",
            block_idx=result.chunk.chunk_id,
            reference=decoded.reference,
            metadata=decoded.transport_metadata,
        )

    def source_progress(self, owned_frame_end, source_records):
        def send():
            self.remote.source_progress(
                owned_frame_end=owned_frame_end, source_records=source_records
            )

        if self.parallel is None:
            send()
        else:
            # Unlike append/pop, this metadata-only call has no existing
            # collective error propagation. Do not strand peers on send failure.
            self.parallel.leader_call(send)

    def finish(self, total_frames):
        self.remote.finish(total_frames=total_frames)
        self.remote.pop()  # Preserve collective failure propagation on EOS.

    def close(self):
        self.remote.dispose()
        self.provenance = None
