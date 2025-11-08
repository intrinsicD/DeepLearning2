from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nl_mm.modules.optim.d_mgd import DMGD


def test_dmgd_modulation_parameters_update():
    param = torch.nn.Parameter(torch.tensor([1.0]))
    opt = DMGD([param], lr=0.1, beta=0.0, learnable_modulation=True, mlp_lr=0.1)

    before = [p.detach().clone() for p in opt.mlp.parameters()]

    opt.zero_grad()
    loss = 0.5 * param.pow(2).sum()
    loss.backward()
    opt.step()

    after = [p.detach().clone() for p in opt.mlp.parameters()]
    assert any(not torch.allclose(b, a) for b, a in zip(before, after))


def test_dmgd_can_disable_learnable_modulation():
    param = torch.nn.Parameter(torch.tensor([1.0]))
    opt = DMGD([param], lr=0.1, beta=0.0, learnable_modulation=False)

    before = [p.detach().clone() for p in opt.mlp.parameters()]

    opt.zero_grad()
    loss = 0.5 * param.pow(2).sum()
    loss.backward()
    opt.step()

    after = [p.detach().clone() for p in opt.mlp.parameters()]
    assert all(torch.allclose(b, a) for b, a in zip(before, after))
