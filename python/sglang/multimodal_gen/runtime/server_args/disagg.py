# SPDX-License-Identifier: Apache-2.0
"""Disaggregated diffusion server argument helpers."""

from __future__ import annotations

from typing import ClassVar, Literal

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.utils.common import (
    format_tcp_endpoint,
    parse_tcp_host_port,
)
from sglang.multimodal_gen.utils import FlexibleArgumentParser


class DisaggServerArgsMixin:
    DISAGG_RESULT_PORT_OFFSETS: ClassVar[dict[RoleType, int]] = {
        RoleType.ENCODER: 1,
        RoleType.DENOISER: 2,
        RoleType.DECODER: 3,
    }

    encoder_gpus: str | None = None
    denoiser_gpus: str | None = None
    decoder_gpus: str | None = None
    disagg_encoder_port: int | None = None
    disagg_decoder_port: int | None = None
    disagg_encoder_ports: list[int] | None = None
    disagg_decoder_ports: list[int] | None = None
    disagg_max_sessions: int | None = None

    def get_role_parallelism(self, role_type: RoleType) -> dict[str, int | None]:
        _none = {
            "tp_size": None,
            "sp_degree": None,
            "ulysses_degree": None,
            "ring_degree": None,
        }
        if role_type == RoleType.ENCODER:
            return {**_none, "tp_size": self.encoder_tp}
        if role_type == RoleType.DENOISER:
            return {
                "tp_size": self.denoiser_tp,
                "sp_degree": self.denoiser_sp,
                "ulysses_degree": self.denoiser_ulysses,
                "ring_degree": self.denoiser_ring,
            }
        if role_type == RoleType.DECODER:
            return {**_none, "sp_degree": self.decoder_sp}
        return _none

    def derive_pool_result_endpoint(self) -> str:
        host, base_port = parse_tcp_host_port(
            self.disagg_server_addr, "disagg_server_addr"
        )
        role = (
            self.disagg_role
            if isinstance(self.disagg_role, RoleType)
            else RoleType.from_string(self.disagg_role)
        )
        try:
            offset = self.DISAGG_RESULT_PORT_OFFSETS[role]
        except KeyError as exc:
            raise ValueError(
                "pool result endpoints are only defined for encoder, denoiser, "
                f"and decoder roles, got {role.value!r}"
            ) from exc
        return format_tcp_endpoint(host, base_port + offset, "pool_result_endpoint")

    def derive_pool_work_endpoint(self) -> str:
        return format_tcp_endpoint("0.0.0.0", self.scheduler_port, "pool_work_endpoint")

    def derive_pool_control_endpoint(self) -> str:
        return format_tcp_endpoint(
            "0.0.0.0", self.scheduler_port + 1, "pool_control_endpoint"
        )

    def derive_pool_control_advertised_endpoint(self) -> str:
        host = self.host or self.disagg_p2p_hostname or "127.0.0.1"
        if host == "0.0.0.0":
            host = self.disagg_p2p_hostname or "127.0.0.1"
        return format_tcp_endpoint(
            host, self.scheduler_port + 1, "pool_control_advertised_endpoint"
        )

    def resolved_role_device(self) -> Literal["cpu", "cuda"]:
        if self.disagg_role_device == "auto":
            return "cpu" if self.num_gpus <= 0 else "cuda"
        return self.disagg_role_device

    @classmethod
    def add_disagg_cli_args(cls, parser: FlexibleArgumentParser) -> None:
        role_default = (
            cls.disagg_role.value
            if isinstance(cls.disagg_role, RoleType)
            else cls.disagg_role
        )
        parser.add_argument(
            "--disagg-role",
            type=str,
            default=role_default,
            choices=RoleType.choices(),
            help="Role for disaggregated pipeline.",
        )
        parser.add_argument(
            "--disagg-timeout",
            type=int,
            default=cls.disagg_timeout,
            help="Timeout in seconds for pending disagg requests. "
            f"Default: {cls.disagg_timeout}.",
        )
        parser.add_argument(
            "--disagg-downstream-wait-timeout",
            type=int,
            default=cls.disagg_downstream_wait_timeout,
            help="Timeout in seconds while waiting for a downstream role slot. "
            f"Default: {cls.disagg_downstream_wait_timeout}.",
        )
        parser.add_argument(
            "--disagg-dispatch-policy",
            type=str,
            default=cls.disagg_dispatch_policy,
            choices=["round_robin", "max_free_slots"],
            help="Dispatch policy for pool mode disagg routing.",
        )
        parser.add_argument(
            "--disagg-instance-id",
            type=int,
            default=cls.disagg_instance_id,
            help="Stable per-role instance ID used by DiffusionServer registration.",
        )
        parser.add_argument(
            "--disagg-max-slots-per-instance",
            type=int,
            default=cls.disagg_max_slots_per_instance,
            help="Maximum concurrent transfer/computation slots tracked per instance.",
        )
        parser.add_argument(
            "--disagg-transfer-redundancy",
            type=float,
            default=cls.disagg_transfer_redundancy,
            help="Redundancy factor used when sizing transfer buffers from warmup.",
        )
        parser.add_argument(
            "--disagg-role-device",
            type=str,
            default=cls.disagg_role_device,
            choices=["auto", "cpu", "cuda"],
            help=(
                "Per-role device override. 'cpu' is intended for same-machine "
                "encoder roles."
            ),
        )
        parser.add_argument(
            "--disagg-transfer-backend",
            type=str,
            default=cls.disagg_transfer_backend,
            choices=["auto", "mock", "mooncake"],
            help="Transfer backend for multimodal diffusion disaggregation.",
        )
        parser.add_argument(
            "--disagg-transfer-pool-size",
            type=int,
            default=cls.disagg_transfer_pool_size,
            help="Size of the P2P transfer buffer pool in bytes.",
        )
        parser.add_argument(
            "--disagg-transfer-pin-memory",
            type=str,
            default=cls.disagg_transfer_pin_memory,
            choices=["auto", "off", "required"],
            help="CUDA host-register same-host shared-memory transfer buffers.",
        )
        parser.add_argument(
            "--disagg-p2p-hostname",
            type=str,
            default=cls.disagg_p2p_hostname,
            help="Hostname for P2P transfer engine.",
        )
        parser.add_argument(
            "--disagg-ib-device",
            type=str,
            default=cls.disagg_ib_device,
            help="InfiniBand device for P2P RDMA transfers.",
        )
        parser.add_argument(
            "--disagg-server-addr",
            type=str,
            default=cls.disagg_server_addr,
            help="DiffusionServer head node address for per-role launch mode.",
        )
        parser.add_argument(
            "--encoder-urls",
            type=str,
            default=cls.encoder_urls,
            help="Encoder instance work endpoints for DiffusionServer head mode.",
        )
        parser.add_argument(
            "--denoiser-urls",
            type=str,
            default=cls.denoiser_urls,
            help="Denoiser instance work endpoints for DiffusionServer head mode.",
        )
        parser.add_argument(
            "--decoder-urls",
            type=str,
            default=cls.decoder_urls,
            help="Decoder instance work endpoints for DiffusionServer head mode.",
        )
        parser.add_argument(
            "--encoder-tp",
            type=int,
            default=cls.encoder_tp,
            help="Tensor parallelism for encoder role.",
        )
        parser.add_argument(
            "--denoiser-tp",
            type=int,
            default=cls.denoiser_tp,
            help="Tensor parallelism for denoiser role.",
        )
        parser.add_argument(
            "--denoiser-sp",
            type=int,
            default=cls.denoiser_sp,
            help="Sequence parallelism for denoiser role.",
        )
        parser.add_argument(
            "--denoiser-ulysses",
            type=int,
            default=cls.denoiser_ulysses,
            help="Ulysses SP degree for denoiser role.",
        )
        parser.add_argument(
            "--denoiser-ring",
            type=int,
            default=cls.denoiser_ring,
            help="Ring SP degree for denoiser role.",
        )
        parser.add_argument(
            "--decoder-sp",
            type=int,
            default=cls.decoder_sp,
            help="Sequence parallelism for decoder role.",
        )
        parser.add_argument(
            "--encoder-gpus",
            type=str,
            default=cls.encoder_gpus,
            help="Comma-separated indices into caller's CUDA_VISIBLE_DEVICES for encoder role.",
        )
        parser.add_argument(
            "--denoiser-gpus",
            type=str,
            default=cls.denoiser_gpus,
            help="Comma-separated indices into caller's CUDA_VISIBLE_DEVICES for denoiser role.",
        )
        parser.add_argument(
            "--decoder-gpus",
            type=str,
            default=cls.decoder_gpus,
            help="Comma-separated indices into caller's CUDA_VISIBLE_DEVICES for decoder role.",
        )
        parser.add_argument(
            "--disagg-encoder-port",
            type=int,
            default=cls.disagg_encoder_port,
            help="External disaggregated encoder control port.",
        )
        parser.add_argument(
            "--disagg-decoder-port",
            type=int,
            default=cls.disagg_decoder_port,
            help="External disaggregated decoder control port.",
        )
        parser.add_argument(
            "--disagg-encoder-ports",
            type=int,
            nargs="+",
            default=cls.disagg_encoder_ports,
            help="Candidate pool of external disaggregated encoder control ports.",
        )
        parser.add_argument(
            "--disagg-decoder-ports",
            type=int,
            nargs="+",
            default=cls.disagg_decoder_ports,
            help="Candidate pool of external disaggregated decoder control ports.",
        )
        parser.add_argument(
            "--disagg-max-sessions",
            type=int,
            default=cls.disagg_max_sessions,
            help="Maximum concurrent sessions per pooled encoder/decoder worker.",
        )

    def _validate_disagg_gpu_roles(self) -> None:
        """Validate explicit disaggregated role GPU placements and topologies."""
        from sglang.multimodal_gen.runtime.server_args.server_args import parse_gpu_list

        encoder = parse_gpu_list(self.encoder_gpus)
        denoiser = parse_gpu_list(self.denoiser_gpus)
        decoder = parse_gpu_list(self.decoder_gpus)

        if encoder is None and denoiser is None and decoder is None:
            return

        if encoder is not None or decoder is not None:
            if denoiser is None:
                raise ValueError(
                    "Disaggregated encoder or decoder roles require --denoiser-gpus to be specified"
                )

        has_external_disagg_ports = bool(
            getattr(self, "disagg_encoder_port", None)
            or getattr(self, "disagg_decoder_port", None)
            or getattr(self, "disagg_encoder_ports", None)
            or getattr(self, "disagg_decoder_ports", None)
            or getattr(self, "disagg_max_sessions", None)
        )
        if (
            denoiser is not None
            and encoder is None
            and decoder is None
            and not has_external_disagg_ports
        ):
            raise ValueError(
                "--denoiser-gpus specified without --encoder-gpus or --decoder-gpus"
            )

        if getattr(self, "nnodes", 1) != 1:
            raise ValueError(
                "Disaggregated role worker topology supports a single node only (nnodes == 1)"
            )

        if len(denoiser) != getattr(self, "num_gpus", len(denoiser)):
            raise ValueError(
                f"--denoiser-gpus must list exactly num_gpus (--num-gpus={getattr(self, 'num_gpus', None)}) devices, got {len(denoiser)}"
            )

        if hasattr(self, "ulysses_degree") or hasattr(self, "ring_degree"):
            ulysses = int(getattr(self, "ulysses_degree", 1) or 1)
            ring = int(getattr(self, "ring_degree", 1) or 1)
            tp = int(getattr(self, "tp_size", 1) or 1)
            dp = int(getattr(self, "dp_size", 1) or 1)
            cfg = int(getattr(self, "cfg_parallel_degree", 1) or 1)
            if ulysses * ring * tp * dp * cfg != len(denoiser):
                raise ValueError(
                    f"--denoiser-gpus (denoiser_gpus) count ({len(denoiser)}) must match denoiser parallelism "
                    f"(ulysses_degree={ulysses}, ring_degree={ring})"
                )

        if len(set(denoiser)) != len(denoiser):
            raise ValueError(
                "--denoiser-gpus lists a device more than once (duplicate GPU)"
            )

        if encoder is not None:
            if len(set(encoder)) != len(encoder):
                raise ValueError(
                    "--encoder-gpus lists a device more than once (duplicate GPU)"
                )
            if set(encoder) & set(denoiser):
                raise ValueError(
                    "A GPU cannot belong to both the encoder and denoiser (overlap)"
                )

        if decoder is not None:
            if len(set(decoder)) != len(decoder):
                raise ValueError(
                    "--decoder-gpus lists a device more than once (duplicate GPU)"
                )
            if set(decoder) & set(denoiser):
                raise ValueError(
                    "A GPU cannot belong to both the denoiser and decoder (overlap)"
                )

        if encoder is not None and decoder is not None:
            if set(encoder) & set(decoder):
                raise ValueError(
                    "A GPU cannot belong to both the encoder and decoder (overlap)"
                )
