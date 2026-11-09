"""Run old research dependencies in an isolated Python, never replace Kaggle's kernel packages."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

PYTHON_VERSION = "3.8.5"
UV_VERSION = "0.9.3"
TORCH_VERSION = "2.1.0"
VISION_VERSION = "0.16.0"
TORCH_INDEX = "https://download.pytorch.org/whl/cu118"
MMCV_WHEEL_SHA256 = "9733cbc2cf656e2d64780c93286847740c03c5da3c85ce531ae779d81f01c93c"
MMCV_WHEEL = "https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/mmcv-2.2.0-cp38-cp38-manylinux1_x86_64.whl#sha256=" + MMCV_WHEEL_SHA256


def setup_runtime(launcher, runtime_root, world_size):
    if sys.platform != "linux":
        raise RuntimeError("This runtime installer targets Linux/Kaggle; use the existing CPU test environment locally.")
    if world_size not in (1, 2):
        raise ValueError("Expected one or two GPUs.")
    launcher, runtime_root = Path(launcher), Path(runtime_root)
    runtime_root.mkdir(parents=True, exist_ok=True)
    tool = runtime_root / "uv_tool"
    env = dict(os.environ)
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--target", str(tool),
                    f"uv=={UV_VERSION}"], check=True, env=env)
    uv_env = dict(env, PYTHONPATH=str(tool), UV_CACHE_DIR=str(runtime_root / "uv_cache"),
                  UV_PYTHON_INSTALL_DIR=str(runtime_root / "managed_python"))
    python = runtime_root / "venv/bin/python"
    if not python.exists():
        subprocess.run([sys.executable, "-m", "uv", "venv", "--python", PYTHON_VERSION,
                        "--managed-python", "--seed", str(runtime_root / "venv")], check=True, env=uv_env)
    print("Installing isolated PyTorch CUDA/runtime dependencies; Kaggle kernel is unchanged.", flush=True)
    subprocess.run([str(python), "-m", "pip", "install", "--quiet", f"torch=={TORCH_VERSION}",
                    f"torchvision=={VISION_VERSION}", "--index-url", TORCH_INDEX], check=True, env=env)
    requirements = launcher / "requirements-kaggle.txt"
    subprocess.run([str(python), "-m", "pip", "install", "--quiet", "-r", str(requirements)], check=True, env=env)
    # Exact published MMCV version, prebuilt official wheel; never silently compile or substitute lite.
    subprocess.run([str(python), "-m", "pip", "install", "--quiet", "--no-deps", MMCV_WHEEL], check=True, env=env)
    # Clean up the unused wheel left by an earlier failed setup in this same venv.
    subprocess.run([str(python), "-m", "pip", "uninstall", "-y", "decord"], check=True, env=env)
    checked = subprocess.run([str(python), "-m", "pip", "check"], text=True, capture_output=True, env=env)
    print(checked.stdout, flush=True)
    print(checked.stderr, file=sys.stderr, flush=True)
    (launcher / 'pip_check.txt').write_text(checked.stdout + checked.stderr, encoding='utf-8')
    checked.check_returncode()
    probe = f'''
import importlib.metadata as m, json, sys, torch, torchvision, mmcv, mmengine
assert sys.version_info[:3] == (3, 8, 5), sys.version
assert torch.__version__ == '2.1.0+cu118', torch.__version__
assert torchvision.__version__ == '0.16.0+cu118', torchvision.__version__
assert torch.version.cuda == '11.8', torch.version.cuda
assert mmcv.__version__ == '2.2.0' and mmengine.__version__ == '0.10.7'
try:
    m.version('mmcv-lite')
except m.PackageNotFoundError:
    pass
else:
    raise RuntimeError('Do not mix mmcv and mmcv-lite.')
assert torch.cuda.is_available() and torch.cuda.device_count() >= {world_size}, 'Select GPU T4 x2; CUDA initialization failed.'
from mmcv.ops import nms
boxes = torch.tensor([[0.,0.,10.,10.],[1.,1.,9.,9.]], device='cuda')
_, keep = nms(boxes, torch.tensor([.9,.8], device='cuda'), .5)
assert keep.tolist() == [0], 'Full MMCV CUDA extension test failed.'
print('ADALEA_RUNTIME_JSON=' + json.dumps(dict(python=sys.version, torch=torch.__version__,
    torchvision=torchvision.__version__, cuda=torch.version.cuda,
    gpus=[torch.cuda.get_device_name(i) for i in range({world_size})],
    packages={{n:m.version(n) for n in ['numpy','pandas','scikit-learn','scipy','mmengine','mmcv','opencv-python']}},
    mmcv_cuda_ops_verified=True,
    installed_packages={{d.metadata['Name'].lower(): d.version for d in m.distributions()}})))
'''
    result = subprocess.run([str(python), "-c", probe], text=True, capture_output=True, env=env)
    if result.returncode:
        print(result.stdout, flush=True)
        print(result.stderr, file=sys.stderr, flush=True)
        result.check_returncode()
    report = json.loads(next(line.split("=", 1)[1] for line in result.stdout.splitlines()
                             if line.startswith("ADALEA_RUNTIME_JSON=")))
    report["kernel_python"] = sys.version
    report["python_executable"] = str(python)
    report["contract"] = dict(python=PYTHON_VERSION, torch="2.1.0+cu118", torchvision="0.16.0+cu118", cuda="11.8",
                              mmengine="0.10.7", mmcv="2.2.0", mmaction2="1.2.0-author-source",
                              mmcv_wheel_sha256=MMCV_WHEEL_SHA256,
                              installed_packages=report.get('installed_packages', {}),
                              requirements_sha256=hashlib.sha256(requirements.read_bytes()).hexdigest())
    report['software_provenance'] = {'reported_readme': ['python=3.8.5', 'mmengine=0.10.7', 'mmcv=2.2.0', 'mmaction2=1.2.0'],
        'inferred_not_author_reported': ['torch=2.1.0', 'torchvision=0.16.0', 'cuda=11.8', 'other package pins']}
    report['unused_optional_dependency_omitted'] = 'decord: Nexar uses OpenCV NativeJpegDecode; not DecordInit/Decode'
    freeze = subprocess.run([str(python), '-m', 'pip', 'freeze'], check=True, text=True, capture_output=True, env=env)
    (launcher / 'runtime_pip_freeze.txt').write_text(freeze.stdout, encoding='utf-8')
    (launcher / "runtime_bootstrap.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--launcher", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--world-size", type=int, default=2)
    args = parser.parse_args()
    setup_runtime(args.launcher, args.runtime_root, args.world_size)
