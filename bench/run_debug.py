"""Debug session: fused GDN kernel dump (DBG=2) vs torch reference; prefill phase timing k0 vs k2."""
import os, sys, time, json
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.common import *
from qwen35_fast import gdn_step as G
from qwen35_fast.engine import Engine
torch.manual_seed(0); dev = "cuda"; H, D = 16, 128; C = 3 * H * D
qkv = torch.randn(C, device=dev).bfloat16() * 2; z = torch.randn(H * D, device=dev).bfloat16()
b = torch.randn(H, device=dev).bfloat16(); a = torch.randn(H, device=dev).bfloat16()
cs = torch.randn(C, 3, device=dev).bfloat16(); cw = (torch.randn(C, 1, 4, device=dev) * 0.5).bfloat16()
neg_expA = -torch.rand(H, device=dev).mul(4).exp(); dt_bias = torch.randn(H, device=dev).bfloat16()
gn = (1 + 0.1 * torch.randn(D, device=dev)).bfloat16(); S = torch.randn(H, D, D, device=dev) * 0.3
xc_all = torch.cat([cs[None], qkv[None, :, None]], -1); y = F.silu(F.conv1d(xc_all, cw, groups=C))[0].T
qf, kf, vf = [t.reshape(H, D).float() for t in y.split([H * D] * 3, -1)]
kn = kf / torch.sqrt((kf * kf).sum(-1, keepdim=True) + 1e-6)
beta = b.sigmoid().float(); g = neg_expA * F.softplus(a.float() + dt_bias)
Sd = S * g.exp()[:, None, None]; kv_ref = (Sd * kn[:, :, None]).sum(1)
dbg = torch.empty(3 * H * D, device=dev, dtype=torch.float32)
G.gdn_step(qkv, z, b, a, cs.clone(), cw, neg_expA, dt_bias, gn, S.clone(), dbg, dbg=2); dbg = dbg.reshape(H, 3, D)
print("exp(g) kernel vs ref (head-wise max diff):", (dbg[:, 0, 0] - g.exp()).abs().max().item(), "kernel", dbg[:3, 0, 0].tolist(), "ref", g.exp()[:3].tolist())
print("beta   kernel vs ref:", (dbg[:, 1, 0] - beta).abs().max().item(), "kernel", dbg[:3, 1, 0].tolist(), "ref", beta[:3].tolist())
print("kv     kernel vs ref:", (dbg[:, 2] - kv_ref).abs().max().item(), "ref mag", kv_ref.abs().max().item(), flush=True)
try:
    G.test()
except Exception as e:
    print("gdn test:", e)
# prefill phase timing
path = model_path(); wl = workloads("dev", [512])
for k in (0, 2):
    eng = Engine(path, spec_k=k, compile_blocks=True); eng.ensure_graph()
    for _ in range(3): eng.generate(wl[0]["ids"], 8, ignore_eos=True)
    ids = wl[0]["ids"]; ts = []
    for _ in range(5):
        torch.cuda.synchronize(); t0 = time.perf_counter(); eng.reset(); torch.cuda.synchronize(); t1 = time.perf_counter()
        tok = torch.tensor(ids, device=dev); pos = torch.arange(len(ids), device=dev)
        with torch.no_grad(): hn, _ = eng._body(tok, pos, prefill=True)
        torch.cuda.synchronize(); t2 = time.perf_counter()
        gg = eng._argmax(hn[-1:]); v = gg.item(); t3 = time.perf_counter()
        r = eng.generate(ids, 8, ignore_eos=True); t4 = r["ttft_s"]
        ts.append((t1 - t0, t2 - t1, t3 - t2, t4))
    print(f"k={k} reset/body/argmax+item/generate-ttft ms:", [tuple(round(1000 * x, 2) for x in t) for t in ts[-2:]], flush=True)
    del eng; torch.cuda.empty_cache()
