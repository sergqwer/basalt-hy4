"""Sanity: GPTQ vs RTN output error on real layer-1 gate_up weights with correlated synthetic inputs (CPU)."""
import sys, time, torch
sys.path.insert(0, r"D:\Projects\hy4-data\gptq\tools")
import gptq_hy4 as Gq
torch.manual_seed(0)
dev = "cpu"; tab = Gq.ue4m3_table(dev)
W = Gq.Bf16().get(1, "gate_up_proj", 0, 2).float()
B, rows, K = W.shape
A = torch.randn(K, K) / K ** 0.5; S = torch.linspace(3, 0.05, K)
X = (torch.randn(20000, K) * S) @ A          # anisotropic inputs
H = (X.T @ X / X.shape[0]).expand(B, K, K).contiguous()
gs = torch.cat([(W[:, :1536].abs().amax((1, 2)) / 2688).unsqueeze(1).expand(B, 1536), (W[:, 1536:].abs().amax((1, 2)) / 2688).unsqueeze(1).expand(B, 1536)], 1).contiguous()
t = time.time(); _, _, Qr = Gq.quant_rtn(W, gs, tab); _, _, Qg = Gq.quant_gptq(W, H, gs, tab); print("time %.1fs" % (time.time() - t))
Xt = (torch.randn(5000, K) * S) @ A
for nm, Q in (("rtn", Qr), ("gptq", Qg)):
    e = sum(float(((Xt @ (Q[j] - W[j]).T) ** 2).sum() / ((Xt @ W[j].T) ** 2).sum()) for j in range(B)) / B
    print(nm, "rel out err %.5f" % e)
