# Copyright 2026 AlphaHebe contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor
from transformers import AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM


@pytest.fixture
def process_group(tmp_path):
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'dist'}", rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


def test_rollout_sync_preserves_updates_smaller_than_bf16_resolution(monkeypatch, process_group):
    from verl.workers.engine.fsdp import transformer_impl as impl

    model = torch.nn.Linear(4, 4, bias=False)
    weight = torch.full((4, 4), 1.0 + 2e-7)
    shard = distribute_tensor(weight, init_device_mesh("cpu", (1,)), [Shard(0)])
    monkeypatch.setattr(model, "state_dict", lambda: {"weight": shard})
    monkeypatch.setattr(impl, "load_fsdp_model_to_gpu", lambda module: None)
    monkeypatch.setattr(impl, "get_device_id", lambda: "cpu")
    engine = SimpleNamespace(
        module=model,
        model_config=SimpleNamespace(lora={}),
        _uses_fsdp2_cpu_offload_policy=False,
        _is_offload_param=False,
        _qat_enabled=False,
    )
    weights, peft = impl.FSDPEngine.get_per_tensor_param(engine)
    actual = dict(weights)["weight"]
    assert peft is None
    assert actual.dtype == torch.float32
    assert not torch.equal(weight, weight.bfloat16().float())
    torch.testing.assert_close(actual, weight, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_hf_export_auto_reload_preserves_weight_dtype_and_updates(tmp_path, monkeypatch, process_group, dtype):
    from verl.utils.checkpoint import fsdp_checkpoint_manager as impl

    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        tie_word_embeddings=True,
    )
    config.architectures = ["Qwen3ForCausalLM"]
    model = Qwen3ForCausalLM(config).to(dtype=dtype)
    with torch.no_grad():
        model.model.embed_tokens.weight.fill_(1.0 + 2e-7)
    monkeypatch.setattr(impl, "get_fsdp_state_ctx", lambda *args: nullcontext())
    monkeypatch.setattr(impl, "fsdp_version", lambda module: 2)
    monkeypatch.setattr(impl, "get_fsdp_full_state_dict", lambda module, **kwargs: module.state_dict())
    manager = impl.FSDPCheckpointManager(
        model, checkpoint_config=OmegaConf.create({"save_contents": ["hf_model"], "load_contents": []})
    )
    manager.save_checkpoint(str(tmp_path / "checkpoint"), global_step=1)
    loaded = AutoModelForCausalLM.from_pretrained(tmp_path / "checkpoint/huggingface", dtype="auto")
    assert loaded.dtype == dtype
    for name, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value, rtol=0, atol=0)
