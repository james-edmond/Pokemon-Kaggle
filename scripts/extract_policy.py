"""Extract the policy-only state_dict from a training checkpoint.
Usage: python scripts/extract_policy.py runs/phase3-a/checkpoint-0120.pt submission_src/policy.pt"""
import os
import sys

import torch


def main():
    src, dst = sys.argv[1], sys.argv[2]
    ck = torch.load(src, map_location="cpu", weights_only=False)
    sd = ck["policy"] if isinstance(ck, dict) and "policy" in ck else ck
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    torch.save({k: v.cpu() for k, v in sd.items()}, dst)
    print(f"wrote {dst} ({len(sd)} tensors)")


if __name__ == "__main__":
    main()
