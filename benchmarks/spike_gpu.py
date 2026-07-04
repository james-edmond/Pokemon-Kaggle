"""GTX 1060 spike: matmul+backward on cuda, encoder fwd/bwd timing, suite compat."""
import time

import torch

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("torch", torch.__version__, "| device", dev,
      "| arch", torch.cuda.get_arch_list() if dev == "cuda" else "-")

x = torch.randn(512, 512, device=dev, requires_grad=True)
(x @ x).sum().backward()
assert torch.isfinite(x.grad).all()
print("matmul+backward ok")

from ptcg.cards import build_tables
from ptcg.model import Encoder, student_config

enc = Encoder(student_config(build_tables())).to(dev)
batch = {
    "card": torch.zeros(256, 192, dtype=torch.int64, device=dev),
    "numeric": torch.randn(256, 192, 40, device=dev),
    "owner": torch.zeros(256, 192, dtype=torch.int64, device=dev),
    "zone": torch.zeros(256, 192, dtype=torch.int64, device=dev),
    "kind": torch.zeros(256, 192, dtype=torch.int64, device=dev),
    "pos": torch.zeros(256, 192, dtype=torch.int64, device=dev),
    "mask": torch.ones(256, 192, dtype=torch.bool, device=dev),
}
for _ in range(2):  # warmup
    enc(batch).sum().backward()
if dev == "cuda":
    torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(5):
    enc(batch).sum().backward()
if dev == "cuda":
    torch.cuda.synchronize()
print(f"student encoder fwd+bwd b256: {(time.perf_counter() - t0) / 5 * 1000:.0f} ms")
