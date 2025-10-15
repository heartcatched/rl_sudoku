# pv_mcts_agent.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def _make_conv_from_weight(W, b):
    out_c, in_c, kh, kw = W.shape
    pad = (kh // 2, kw // 2)
    conv = nn.Conv2d(in_c, out_c, kernel_size=(kh, kw), stride=1, padding=pad)
    with torch.no_grad():
        conv.weight.copy_(W)
        if b is not None:
            conv.bias.copy_(b)
        else:
            nn.init.zeros_(conv.bias)
    return conv

def _make_linear_from_weight(W, b):
    out_f, in_f = W.shape
    lin = nn.Linear(in_f, out_f)
    with torch.no_grad():
        lin.weight.copy_(W)
        if b is not None:
            lin.bias.copy_(b)
        else:
            nn.init.zeros_(lin.bias)
    return lin

class PVNetDynamic(nn.Module):
    """
    Собирает trunk, head_policy, head_value по state_dict, автоматически определяя Conv2d/Linear.
    Между Conv2d и Linear вставляет Flatten.
    ReLU ставится после каждого веса-слоя, кроме последнего в голове value (обычно регрессия).
    """
    def __init__(self, sd: dict):
        super().__init__()
        self.sd = sd
        self.trunk = self._build_sequential(prefix="trunk")
        self.head_policy = self._build_sequential(prefix="head_policy")
        self.head_value  = self._build_sequential(prefix="head_value", last_relu=False)

    def _collect_layers(self, prefix):
        idxs = set()
        pfx = prefix + "."
        for k in self.sd.keys():
            if k.startswith(pfx) and k.endswith(".weight"):
                try:
                    i = int(k[len(pfx):].split(".")[0])
                    idxs.add(i)
                except:
                    pass
        return sorted(list(idxs))

    def _build_sequential(self, prefix, last_relu=True):
        layers = []
        idxs = self._collect_layers(prefix)
        prev_was_conv = False
        for i, idx in enumerate(idxs):
            W = self.sd.get(f"{prefix}.{idx}.weight", None)
            b = self.sd.get(f"{prefix}.{idx}.bias",   None)
            if W is None:
                continue

            if W.ndim == 4:
                mod = _make_conv_from_weight(W, b)
                layers.append(mod)
                prev_was_conv = True
            elif W.ndim == 2:
                if prev_was_conv:
                    layers.append(nn.Flatten())
                    prev_was_conv = False
                mod = _make_linear_from_weight(W, b)
                layers.append(mod)
            else:
                raise ValueError(f"Unsupported weight dim for {prefix}.{idx}.weight: {W.shape}")

            is_last = (i == len(idxs) - 1)
            if not is_last:
                layers.append(nn.ReLU())
            else:
                if prefix == "trunk" and last_relu:
                    layers.append(nn.ReLU())
                if prefix == "head_policy" and last_relu:
                    layers.append(nn.ReLU())

        return nn.Sequential(*layers) if layers else nn.Identity()

    def _head_expects_conv(self, head: nn.Sequential) -> bool:
        return (len(head) > 0) and isinstance(head[0], nn.Conv2d)

    def forward(self, x):
        f = self.trunk(x)  


        if self._head_expects_conv(self.head_policy):
            p_in = f  
        else:
            p_in = f.flatten(1) if f.ndim == 4 else f 

        if self._head_expects_conv(self.head_value):
            v_in = f  
        else:
            v_in = f.flatten(1) if f.ndim == 4 else f  

        p = self.head_policy(p_in)
        v = self.head_value(v_in)
        return p, v

class PVAgentCompat:
    def __init__(self, device, state_dict):
        self.device=device
        self.net=PVNetDynamic(state_dict).to(device)
        self.net.eval()

    @torch.no_grad()
    def policy_logits(self,state_tensor):
        p,_=self.net(state_tensor)
        return p.squeeze(0)

    @torch.no_grad()
    def act(self,state_tensor,legal_idx):
        logits=self.policy_logits(state_tensor)
        mask=torch.full_like(logits,-float('inf'))
        if len(legal_idx)>0:
            mask[torch.tensor(legal_idx,device=logits.device)]=0
        return int(torch.argmax(logits+mask).item())
