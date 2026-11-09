"""DiT entrypoint for an already allocated and partitioned E1+SP2+D1 job."""

import multiprocessing as mp
import os
import sys


def main():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(visible) != 2 or len(set(visible)) != 2 or not all(visible):
        raise ValueError("DiT requires two distinct pre-partitioned GPU devices")
    mp.set_start_method("spawn", force=True)
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.encoder_worker import (
        initialize_runtime_imports,
    )

    initialize_runtime_imports()
    from sglang.multimodal_gen.runtime.launch_server import dispatch_launch
    from sglang.multimodal_gen.runtime.server_args.server_args import (
        prepare_server_args,
    )
    from sglang.multimodal_gen.runtime.utils.common import kill_process_tree

    args = prepare_server_args(sys.argv[1:])
    if args.num_gpus != 2 or any(
        (args.encoder_gpus, args.denoiser_gpus, args.decoder_gpus)
    ):
        raise ValueError("The parent launcher owns role allocation")
    args.disagg_encoder_port = 31001
    args.disagg_decoder_port = 31003
    try:
        dispatch_launch(args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
