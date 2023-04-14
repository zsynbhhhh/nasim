import os.path as osp

from .generated import AVAIL_GEN_BENCHMARKS

BENCHMARK_DIR = osp.dirname(osp.abspath(__file__))

AVAIL_STATIC_BENCHMARKS = {
    "tiny": {
        "file": osp.join(BENCHMARK_DIR, "tiny.yaml"),
        "name": "tiny",
        "step_limit": 1000,
        "max_score": 195
    },
    "tiny-hard": {
        "file": osp.join(BENCHMARK_DIR, "tiny-hard.yaml"),
        "name": "tiny-hard",
        "step_limit": 1000,
        "max_score": 192
    },
    "tiny-small": {
        "file": osp.join(BENCHMARK_DIR, "tiny-small.yaml"),
        "name": "tiny-small",
        "step_limit": 1000,
        "max_score": 189
    },
    "small": {
        "file": osp.join(BENCHMARK_DIR, "small.yaml"),
        "name": "small",
        "step_limit": 1000,
        "max_score": 186
    },
    "small-honeypot": {
        "file": osp.join(BENCHMARK_DIR, "small-honeypot.yaml"),
        "name": "small-honeypot",
        "step_limit": 1000,
        "max_score": 186
    },
    "small-linear": {
        "file": osp.join(BENCHMARK_DIR, "small-linear.yaml"),
        "name": "small-linear",
        "step_limit": 1000,
        "max_score": 187
    },
    "medium": {
        "file": osp.join(BENCHMARK_DIR, "medium.yaml"),
        "name": "medium",
        "step_limit": 2000,
        "max_score": 190
    },
    "medium-single-site": {
        "file": osp.join(BENCHMARK_DIR, "medium-single-site.yaml"),
        "name": "medium-single-site",
        "step_limit": 2000,
        "max_score": 195
    },
    "medium-multi-site": {
        "file": osp.join(BENCHMARK_DIR, "medium-multi-site.yaml"),
        "name": "medium-multi-site",
        "step_limit": 2000,
        "max_score": 190
    },
    "large": {
        "file": osp.join(BENCHMARK_DIR, "large.yaml"),
        "name": "large",
        "step_limit": 6000,
        "max_score": 195
    },
    "large-single-site": {
        "file": osp.join(BENCHMARK_DIR, "large-single-site.yaml"),
        "name": "large-single-site",
        "step_limit": 6000
    },
    "large-multi-site": {
        "file": osp.join(BENCHMARK_DIR, "large-multi-site.yaml"),
        "name": "large-multi-site",
        "step_limit": 8000
    },
    "huge": {
        "file": osp.join(BENCHMARK_DIR, "huge.yaml"),
        "name": "huge",
        "step_limit": 10000,
        "max_score": 195
    }



}

AVAIL_BENCHMARKS = list(AVAIL_STATIC_BENCHMARKS.keys()) \
                    + list(AVAIL_GEN_BENCHMARKS.keys())
