import glob
import os

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

try:
    from modules.dataset.augmentation import AugmentationPipe
except ModuleNotFoundError as e:
    raise SystemExit(f"{e}: run with the training env, e.g. "
                     "/home/magic/work_task/xfeat_train_env/bin/python -m finetune.train_distill ...")
from modules.training.utils import get_corresponding_pts, make_batch

MIN_CORRS = 30
DIFFICULTY = 0.10


def seed_worker(worker_id):
    np.random.seed(torch.initial_seed() % 2**32)


class SyntheticPairs:
    def __init__(self, args, batch_size, device):
        self.aug = AugmentationPipe(
            img_dir=args.synthetic_root, device=device, load_dataset=False,
            batch_size=batch_size, out_resolution=args.training_res,
            warp_resolution=args.training_res, sides_crop=0.1,
            max_num_imgs=args.synthetic_max_imgs, num_test_imgs=args.val_images,
            photometric=True, geometric=True, reload_step=4_000)
        files = sorted(self.aug.all_imgs)
        # load_imgs reshuffles all_imgs, so held-out images must not stay in that list
        self.aug.all_imgs, val_files = files[:-args.val_images], files[-args.val_images:]
        if len(self.aug.all_imgs) < args.synthetic_max_imgs + args.val_images:
            raise RuntimeError(f"not enough images in {args.synthetic_root}: "
                               f"{len(self.aug.all_imgs)} for --synthetic-max-imgs {args.synthetic_max_imgs}")
        self.aug.load_imgs()
        self.val_imgs = self._load(val_files)

    def _load(self, files):
        all_imgs, n = self.aug.all_imgs, self.aug.max_num_imgs
        self.aug.all_imgs, self.aug.max_num_imgs = list(files), len(files)
        keep = self.aug.train, self.aug.test
        self.aug.load_imgs()
        val = self.aug.train
        self.aug.train, self.aug.test = keep
        self.aug.all_imgs, self.aug.max_num_imgs = all_imgs, n
        return val

    def next(self, val=False):
        if val:
            train_imgs, self.aug.train = self.aug.train, self.val_imgs
        p1, p2, H1, H2 = make_batch(self.aug, DIFFICULTY)
        if val:
            self.aug.train = train_imgs
        h, w = p1.shape[-2] // 8, p1.shape[-1] // 8
        _, positives = get_corresponding_pts(p1, p2, H1, H2, self.aug, h, w)
        return p1.mean(1, keepdim=True), p2.mean(1, keepdim=True), positives


class MegaDepthPairs:
    def __init__(self, args, batch_size, device, seed):
        from modules.dataset.megadepth.megadepth import MegaDepthDataset
        from modules.dataset.megadepth import megadepth_warper
        self.spvs_coarse = megadepth_warper.spvs_coarse
        self.device = device

        root = args.megadepth_root
        npz = sorted(glob.glob(f"{root}/train_data/megadepth_indices/scene_info_0.1_0.7/*.npz"))
        if len(npz) <= args.val_scenes:
            raise RuntimeError(f"not enough scenes under {root}: {len(npz)}")
        if args.max_scenes:
            npz = npz[:args.max_scenes + args.val_scenes]
        data_dir = f"{root}/MegaDepth_v1"

        def loader(paths, shuffle_seed):
            data = ConcatDataset([MegaDepthDataset(root_dir=data_dir, npz_path=p) for p in paths])
            return DataLoader(data, batch_size=batch_size, shuffle=True,
                              num_workers=args.num_workers, worker_init_fn=seed_worker,
                              generator=torch.Generator().manual_seed(shuffle_seed), drop_last=True)

        self.train_loader = loader(npz[:-args.val_scenes], seed)
        self.val_loader = loader(npz[-args.val_scenes:], args.val_seed)
        self.train_iter, self.val_iter = iter(self.train_loader), None

    def _load(self, d):
        for k in d.keys():
            if isinstance(d[k], torch.Tensor):
                d[k] = d[k].to(self.device)
        positives = self.spvs_coarse(d, 8)
        return d['image0'].mean(1, keepdim=True), d['image1'].mean(1, keepdim=True), positives

    def next(self, val=False):
        if val:
            self.val_iter = self.val_iter or iter(self.val_loader)
            return self._load(next(self.val_iter))
        try:
            d = next(self.train_iter)
        except StopIteration:
            self.train_iter = iter(self.train_loader)
            d = next(self.train_iter)
        return self._load(d)


class PairSource:
    """training_type as in the authors' script: default = 40% synthetic + 60% MegaDepth."""

    def __init__(self, args, device, seed):
        bs, kind = args.batch_size, args.training_type
        self.sources = []
        if kind == "default":
            n_synth = int(bs * 0.4)
            self.sources = [SyntheticPairs(args, n_synth, device),
                            MegaDepthPairs(args, bs - n_synth, device, seed)]
        elif kind == "synthetic":
            self.sources = [SyntheticPairs(args, bs, device)]
        else:
            self.sources = [MegaDepthPairs(args, bs, device, seed)]
        self.skipped = 0

    def _draw(self, val):
        parts = [s.next(val) for s in self.sources]
        p1 = torch.cat([p[0] for p in parts], dim=0)
        p2 = torch.cat([p[1] for p in parts], dim=0)
        positives = [q for p in parts for q in p[2]]
        return p1, p2, positives

    def next(self, val=False):
        """Batches with too few correspondences are dropped, as in train.py,
        but here they do not count as steps."""
        while True:
            p1, p2, positives = self._draw(val)
            if min(len(p) for p in positives) >= MIN_CORRS:
                return p1, p2, positives
            self.skipped += 1
