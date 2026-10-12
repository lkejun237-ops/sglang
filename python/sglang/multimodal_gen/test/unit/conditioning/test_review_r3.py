# SPDX-License-Identifier: Apache-2.0
"""Applicable pre-RPC regressions retained under the approved v6 redesign."""


def test_local_mode_preserves_native_token_range(monkeypatch):
    import ast
    import subprocess
    from types import SimpleNamespace

    import torch
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        codecs as module,
    )

    rows = 16385
    ids = torch.zeros(rows, dtype=torch.int64)

    def presentation(*args, **kwargs):
        return ids, ids

    calls = []

    def encode(ids, **kwargs):
        calls.append(ids.numel())
        return torch.zeros(1, dtype=torch.bfloat16).expand(rows, 5120)

    obj = object.__new__(module.ZingH3Codecs)
    obj.conditioning_mode = "local"
    obj.modules = dict(
        tokenizer=object(),
        processor=SimpleNamespace(image_processor=SimpleNamespace(merge_size=2)),
        text_encoder=SimpleNamespace(encode_ids=encode),
    )
    obj.runtime = {}
    source = subprocess.check_output(
        [
            "git",
            "show",
            "b1899c0bd538a342056a39893e8ce048b17afa32:python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/zing_h3/codecs.py",
        ],
        text=True,
    )
    tree = ast.parse(source)
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ZingH3Codecs"
    )
    old = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "encode_condition"
    )
    scope = {"minimax_h3_ref2va_video_presentation": presentation}
    exec(
        compile(
            ast.Module(body=[old], type_ignores=[]), "<frozen HEAD native>", "exec"
        ),
        scope,
    )
    assert (
        scope["encode_condition"](obj, "long valid prompt", [])["context"].shape[0]
        == rows
    )
    monkeypatch.setattr(module, "minimax_h3_ref2va_video_presentation", presentation)
    print("LOCAL_NATIVE_BASELINE_ROWS", calls)
    result = obj.encode_condition("long valid prompt", [])
    assert result["context"].shape[0] == rows
