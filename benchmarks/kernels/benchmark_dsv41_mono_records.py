# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Time the validated Mono layer launch pairs with fixed graph replay counts."""

import argparse
import json
import os
from functools import partial
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

from tests.models.test_dsv41_mono_numerics import _mono_numerics_body
from vllm.distributed import get_tp_group
from vllm.models.deepseek_v41.amd.mono.runner import DSV41MonoLayer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", default="584/584")
    parser.add_argument("--output", required=True)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    rank, tp = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    records = tuple(map(int, args.records.split("/")))
    original = DSV41MonoLayer.forward
    calls = {}

    def capture_call(self, *a, **kw):
        result = original(self, *a, **kw)
        calls[(a[0].attn.ratio, a[1].shape[0])] = partial(original, self, *a, **kw)
        return result

    with patch.object(DSV41MonoLayer, "forward", capture_call):
        _mono_numerics_body(
            pytest.MonkeyPatch(), tp, 1, rank,
            int(os.environ["MASTER_PORT"]), records,
        )
    group = get_tp_group().cpu_group
    rows = []
    for (ratio, tokens), run in sorted(calls.items()):
        run()
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for _ in range(10):
            graph.replay()
        torch.accelerator.synchronize()
        samples = []
        for _ in range(args.repeats):
            dist.barrier(group=group)
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            for _ in range(args.iterations):
                graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1000 / args.iterations)
        peers = [None] * tp
        dist.all_gather_object(peers, samples, group=group)
        if rank == 0:
            rows.append({"ratio": ratio, "tokens": tokens,
                         "us": [max(p[i] for p in peers) for i in range(args.repeats)]})
    if rank == 0:
        with open(args.output, "w") as output:
            json.dump({"records": records, "tp": tp,
                       "iterations": args.iterations, "results": rows}, output, indent=2)


if __name__ == "__main__":
    main()
