"""Byte check of gptq_hy4's RTN path (torch, CPU) against the RTN pack: layer L, experts e0..e1."""
import sys, numpy as np, torch
sys.path.insert(0, r"D:\Projects\hy4-data\gptq\tools")
import gptq_hy4 as Gq
l, e0, e1 = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
dev = sys.argv[4] if len(sys.argv) > 4 else "cpu"
tab = Gq.ue4m3_table(dev)
bf = Gq.Bf16()
m, eoff, esz, lay = Gq.layout(Gq.__dict__.get("PACK", r"D:/Projects/basalt-work/models/hy4/Hy4-flash-NVFP4.basalt"))
Wgu = bf.get(l, "gate_up_proj", e0, e1).to(dev).float(); Wd = bf.get(l, "down_proj", e0, e1).to(dev).float()
B = e1 - e0; F = Gq.F; L = Gq.L
gs_gu = torch.cat([(Wgu[:, :F].abs().amax((1, 2)) / 2688).unsqueeze(1).expand(B, F), (Wgu[:, F:].abs().amax((1, 2)) / 2688).unsqueeze(1).expand(B, F)], 1).contiguous()
gs_d = (Wd.abs().amax((1, 2)) / 2688).unsqueeze(1).expand(B, L).contiguous()
c, s, _ = Gq.quant_rtn(Wgu, gs_gu, tab); c2, s2, _ = Gq.quant_rtn(Wd, gs_d, tab)
tail = torch.stack([gs_gu[:, 0], gs_gu[:, F], gs_d[:, 0], torch.zeros(B, device=dev)], 1).float().contiguous()
blob = torch.cat([Gq.to_ggml(c, s), Gq.to_ggml(c2, s2), tail.view(torch.uint8).view(B, 16)], 1).cpu().numpy()
off, bl = lay[l]
with open(r"D:/Projects/basalt-work/models/hy4/Hy4-flash-NVFP4.basalt", "rb") as f:
    f.seek(eoff + off + e0 * bl); ref = np.frombuffer(f.read(B * bl), np.uint8).reshape(B, bl)
print("layer", l, "experts", e0, e1, "identical:", bool((ref == blob).all()), "mismatch bytes", int((ref != blob).sum()))
