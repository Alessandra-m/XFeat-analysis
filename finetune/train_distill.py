import argparse
import csv
import hashlib
import json
import os
import platform
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.paths import HERE, XFEAT_DIR

import numpy as np
import torch

from modules.model import XFeatModel
from modules.training.utils import check_accuracy

import benchmark.hpatches_eval as he
from finetune.data import PairSource
from finetune.losses import at_points, distill_losses

WEIGHTS = os.path.join(XFEAT_DIR, "weights", "xfeat.pt")
FROZEN = ("keypoint_head", "fine_matcher")
TERMS = ("cos", "mse", "rel", "cross", "gt")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_teacher(device):
    net = XFeatModel().to(device).eval()
    net.load_state_dict(torch.load(WEIGHTS, map_location=device))
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def load_student(path, device):
    """path: a pruned full-module .pth, or "baseline" for the unpruned control."""
    if path == "baseline":
        net = XFeatModel().to(device)
        net.load_state_dict(torch.load(WEIGHTS, map_location=device))
        return net
    return torch.load(path, map_location=device, weights_only=False)


def freeze(net):
    for name in FROZEN:
        for p in getattr(net, name).parameters():
            p.requires_grad_(False)


def set_train_mode(net, frozen_bn=False):
    """Frozen parts stay in eval: BatchNorm running stats would drift otherwise.
    frozen_bn: every BatchNorm uses its running stats, as the teacher does."""
    net.train()
    for name in FROZEN:
        getattr(net, name).eval()
    if frozen_bn:
        for m in net.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.eval()


def frozen_state(net):
    return {f"{name}.{k}": v.detach().clone() for name in FROZEN
            for k, v in getattr(net, name).state_dict().items()}


def check_frozen(net, before):
    after = frozen_state(net)
    changed = [k for k in before if not torch.equal(before[k], after[k])]
    if changed:
        raise SystemExit(f"frozen modules changed: {changed}")


def check_interface(net, device):
    net.eval()
    with torch.inference_mode():
        f, k, h = net(torch.rand(1, 1, 480, 640, device=device))
    if not (f.shape[1] == 64 and k.shape[1] == 65 and h.shape[1] == 1):
        raise SystemExit("interface broken (feats/kpts/heat shape changed)")


def file_sha256(path):
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def update_fingerprint(h, p1, positives):
    """Tells whether two runs saw the same batches."""
    h.update(np.array([len(p) for p in positives], dtype=np.int64).tobytes())
    h.update((p1[..., ::32, ::32] * 255).round().byte().cpu().numpy().tobytes())


@torch.no_grad()
def evaluate(student, teacher, val_batches, device, frozen_bn):
    """Held-out batches in eval mode, i.e. with BatchNorm stats used at inference."""
    student.eval()
    rows = {"desc_cos": [], "rel_l1": [], "acc_student": [], "acc_teacher": []}
    for p1, p2, positives in val_batches:
        p1, p2 = p1.to(device), p2.to(device)
        s1, s2, t1, t2 = student(p1), student(p2), teacher(p1), teacher(p2)
        rows["desc_cos"].append(torch.cosine_similarity(s1[0], t1[0], dim=1).mean().item())
        rows["rel_l1"].append((s1[2] - t1[2]).abs().mean().item())
        for b, pos in enumerate(positives):
            pos = pos.to(device)
            pts1, pts2 = pos[:, :2], pos[:, 2:]
            rows["acc_student"].append(check_accuracy(at_points(s1[0], b, pts1), at_points(s2[0], b, pts2)))
            rows["acc_teacher"].append(check_accuracy(at_points(t1[0], b, pts1), at_points(t2[0], b, pts2)))
    set_train_mode(student, frozen_bn)
    return {k: float(np.mean(v)) for k, v in rows.items()}


def parse_args():
    p = argparse.ArgumentParser(description="Distillation fine-tuning of pruned XFeat")
    p.add_argument("--student", required=True, help='pruned .pth or "baseline" (unpruned control)')
    p.add_argument("--name", required=True, help="run name, e.g. config_A")
    p.add_argument("--out-dir", default=os.path.join(HERE, "finetune", "results"))
    p.add_argument("--training-type", default="default", choices=["default", "synthetic", "megadepth"])
    p.add_argument("--megadepth-root", default=None)
    p.add_argument("--synthetic-root", default=None)
    p.add_argument("--synthetic-max-imgs", type=int, default=3000)
    p.add_argument("--max-scenes", type=int, default=0, help="MegaDepth subset: first N train scenes, 0 = all")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--val-images", type=int, default=50, help="held-out synthetic images")
    p.add_argument("--val-scenes", type=int, default=2, help="held-out MegaDepth scenes")
    p.add_argument("--val-batches", type=int, default=20)
    p.add_argument("--val-seed", type=int, default=12345, help="fixed for every run")
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--n-steps", type=int, default=10_000, help="optimizer steps")
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--training-res", type=lambda s: tuple(map(int, s.split(','))), default=(800, 608))
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--max-corrs", type=int, default=0,
                   help="random subset of correspondences per pair for the dual-softmax terms, 0 = all")
    for term in TERMS:
        p.add_argument(f"--w-{term}", type=float, default=1.0, help=f"weight of the {term} term")
    p.add_argument("--frozen-bn", action="store_true", help="keep BatchNorm in eval mode while training")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deterministic", action="store_true", help="deterministic CUDA kernels, slower")
    p.add_argument("--save-at", type=int, nargs="*", default=[], help="extra checkpoint steps")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    args = p.parse_args()
    if args.training_type != "megadepth" and not args.synthetic_root:
        p.error("--synthetic-root is required for this training type")
    if args.training_type != "synthetic" and not args.megadepth_root:
        p.error("--megadepth-root is required for this training type")
    return args


def main():
    args = parse_args()
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    torch.backends.cudnn.deterministic = args.deterministic

    run_dir = os.path.join(args.out_dir, f"{args.name}_s{args.seed}")
    os.makedirs(run_dir, exist_ok=True)

    teacher = load_teacher(device)
    student = load_student(args.student, device)
    freeze(student)
    before = frozen_state(student)
    trainable = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=args.lr)

    # validation batches come first, from a fixed seed, so they are the same in every run
    seed_everything(args.val_seed)
    source = PairSource(args, device, args.seed)
    val_batches = []
    for _ in range(args.val_batches):
        p1, p2, positives = source.next(val=True)
        val_batches.append((p1.cpu(), p2.cpu(), [q.cpu() for q in positives]))
    source.skipped = 0
    seed_everything(args.seed)

    config = {"args": {k: list(v) if isinstance(v, tuple) else v for k, v in vars(args).items()},
              "student_sha256": file_sha256(args.student),
              "n_trainable_params": sum(p.numel() for p in trainable),
              "repo_commit": he.git_commit(HERE), "repo_dirty": he.git_dirty(HERE),
              "xfeat_commit": he.git_commit(XFEAT_DIR),
              "python": platform.python_version(), "torch": torch.__version__,
              "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None}

    corr_gen = torch.Generator(device=device).manual_seed(args.seed)
    fingerprint = hashlib.sha256()
    t0 = time.time()
    train_f = open(os.path.join(run_dir, "train_log.csv"), "w", newline="")
    val_f = open(os.path.join(run_dir, "val_log.csv"), "w", newline="")
    train_log = csv.DictWriter(train_f, ["step", "loss", *TERMS, "elapsed_s"])
    val_log = csv.DictWriter(val_f, ["step", "desc_cos", "rel_l1", "acc_student", "acc_teacher"])
    train_log.writeheader()
    val_log.writeheader()

    def validate(step):
        m = evaluate(student, teacher, val_batches, device, args.frozen_bn)
        val_log.writerow({"step": step, **{k: round(v, 5) for k, v in m.items()}})
        val_f.flush()
        print(f"[val {step}] cos {m['desc_cos']:.4f}  rel_l1 {m['rel_l1']:.4f}  "
              f"acc {m['acc_student']:.3f} (teacher {m['acc_teacher']:.3f})", flush=True)
        return m

    set_train_mode(student, args.frozen_bn)
    first = validate(0)
    for step in range(1, args.n_steps + 1):
        p1, p2, positives = source.next()
        update_fingerprint(fingerprint, p1, positives)
        if args.max_corrs:
            # own generator: drawing from the global one would shift the batch stream
            positives = [q[torch.randperm(len(q), device=q.device, generator=corr_gen)[:args.max_corrs]]
                         for q in positives]

        with torch.no_grad():
            t1, t2 = teacher(p1), teacher(p2)
        s1, s2 = student(p1), student(p2)
        terms = distill_losses(s1, s2, t1, t2, positives)
        loss = sum(getattr(args, f"w_{k}") * terms[k] for k in TERMS)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, args.clip)
        opt.step()
        opt.zero_grad()

        if step % args.log_every == 0 or step == 1:
            train_log.writerow({"step": step, "loss": round(loss.item(), 5),
                                **{k: round(terms[k].item(), 5) for k in TERMS},
                                "elapsed_s": round(time.time() - t0, 1)})
            train_f.flush()
        if step % args.val_every == 0:
            validate(step)
        if step in args.save_at:
            torch.save(student, os.path.join(run_dir, f"step_{step}.pth"))

    if args.n_steps % args.val_every:
        validate(args.n_steps)
    train_f.close()
    val_f.close()

    check_frozen(student, before)
    check_interface(student, device)
    torch.save(student.eval(), os.path.join(run_dir, "final.pth"))

    config.update({"batch_fingerprint": fingerprint.hexdigest(), "skipped_batches": source.skipped,
                   "wall_time_s": round(time.time() - t0, 1), "val_start": first})
    with open(os.path.join(run_dir, "run_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"done in {config['wall_time_s']}s, skipped {source.skipped} -> {run_dir}")


if __name__ == "__main__":
    main()
