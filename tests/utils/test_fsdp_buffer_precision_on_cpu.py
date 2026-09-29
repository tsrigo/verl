# Copyright 2026 AlphaHebe contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch
from torch import nn

from verl.utils.fsdp_utils import cast_model_parameters


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_parameter_cast_preserves_rotary_values_and_buffer_contract(dtype):
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    frequencies = torch.tensor([1.0, 0.1234567, 0.00314159, 0.00012345])
    model[0].register_buffer("inv_freq", frequencies, persistent=False)
    model[1].register_buffer("inv_freq", frequencies, persistent=False)
    model.register_buffer("scale", torch.tensor(0.1234567), persistent=True)
    model.register_buffer("counter", torch.tensor(3), persistent=True)
    model.register_buffer("optional", None, persistent=False)
    original_scale = model.scale
    original_weights = model[0].weight.detach().clone()
    positions = torch.arange(8192, dtype=torch.float32)
    expected_cos = torch.cos(positions[:, None] * frequencies)
    assert not torch.equal(expected_cos, torch.cos(positions[:, None] * frequencies.bfloat16().float()))

    assert cast_model_parameters(model, dtype) is model

    assert all(parameter.dtype == dtype for parameter in model.parameters())
    torch.testing.assert_close(model[0].weight, original_weights.to(dtype), rtol=0, atol=0)
    assert model[0].inv_freq is frequencies
    assert model[1].inv_freq is frequencies
    assert model.scale is original_scale
    assert model.counter.dtype == torch.int64
    assert model.optional is None
    assert set(model.state_dict()) == {"0.weight", "0.bias", "1.weight", "1.bias", "scale", "counter"}
    torch.testing.assert_close(torch.cos(positions[:, None] * model[0].inv_freq), expected_cos, rtol=0, atol=0)


def test_parameter_cast_keeps_meta_buffers_and_tied_parameters():
    model = nn.Sequential(nn.Linear(4, 4, device="meta"), nn.Linear(4, 4, device="meta"))
    model[1].weight = model[0].weight
    model[0].register_buffer("inv_freq", torch.empty(4, device="meta", dtype=torch.float32), persistent=False)
    original = model[0].inv_freq

    cast_model_parameters(model, torch.bfloat16)

    assert model[0].weight is model[1].weight
    assert model[0].weight.dtype == torch.bfloat16
    assert model[0].inv_freq is original
    assert original.dtype == torch.float32
    assert original.device.type == "meta"
