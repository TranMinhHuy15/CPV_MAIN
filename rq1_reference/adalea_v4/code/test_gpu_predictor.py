"""Optional real CUDA predictor check, synthetic features; NOT extractor/RTX4080 validation."""
import json
import argparse
from pathlib import Path
import torch
from data import ROOT, save_json
from model import AdaLEA, adaptive_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', help='Optional NEW report path outside the packaged directory.')
    args = parser.parse_args()
    if args.report and ROOT.resolve() in Path(args.report).resolve().parents:
        raise ValueError('Do not overwrite packaged evidence/hashes. Choose an output path outside this package.')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA not available. Do not mark this test passed.')
    cfg = json.loads((ROOT/'config.json').read_text())
    device = torch.device('cuda')
    torch.manual_seed(7)
    model = AdaLEA(cfg['feature_dim'], cfg['embedding'], cfg['hidden'], cfg['dropout']).to(device)
    g = torch.randn(2, 30, 4096, device=device)
    l = torch.randn(2, 30, cfg['max_objects'], 4096, device=device)
    mask = torch.zeros(2, 30, cfg['max_objects'], dtype=torch.bool, device=device)
    mask[0, :, 0] = True  # Include one video with NO detections to test AMP zero-mask.
    frames = torch.arange(30, device=device)[None].repeat(2, 1)*3
    labels, event, fps = torch.tensor([1, 0], device=device), torch.tensor([87, -1], device=device), torch.ones(2, device=device)*30
    optimizer = torch.optim.SGD(model.parameters(), lr=.002, momentum=.9)
    actual_steps = [0]
    original_step = optimizer.step
    def counted_step(*args, **kwargs):
        actual_steps[0] += 1
        return original_step(*args, **kwargs)
    optimizer.step = counted_step
    scaler = torch.cuda.amp.GradScaler(init_scale=1024.)
    before = model.output.weight.detach().clone()
    torch.cuda.reset_peak_memory_stats()
    for micro in range(16):
        with torch.autocast('cuda', dtype=torch.float16):
            logits, attention = model(g, l, mask)
        assert torch.isfinite(logits).all() and torch.isfinite(attention).all()
        assert attention[1].abs().sum().item() == 0
        loss = adaptive_loss(logits, labels, frames, event, fps, 0.)
        assert torch.isfinite(loss)
        scaler.scale(loss/8).backward()
        if (micro+1)%8 == 0:
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 40.)
            assert torch.isfinite(norm), norm
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
    assert not torch.equal(before, model.output.weight)
    assert actual_steps[0] == 2, 'AMP skipped an optimizer update'
    model.eval()
    with torch.no_grad():
        full = model(g, l, mask)[0]
        prefix = model(g[:, :10], l[:, :10], mask[:, :10])[0]
    assert full.dtype == torch.float32 and torch.allclose(prefix, full[:, :10], atol=1e-5, rtol=1e-5)
    result = dict(success=True, gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                  optimizer_steps=actual_steps[0], microbatches=16, empty_object_amp_finite=True,
                  validation_fp32=True, peak_memory_gib=torch.cuda.max_memory_allocated()/(1<<30),
                  synthetic_features_only=True, pretrained_extractor_tested=False, rtx4080_tested=False)
    if args.report:
        save_json(args.report, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
