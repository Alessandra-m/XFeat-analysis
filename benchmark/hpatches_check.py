import argparse
import os
import sys

import numpy as np

EXPECTED_SEQ = 116
EXPECTED_I = 57
EXPECTED_V = 59
IMAGES = [f"{i}.ppm" for i in range(1, 7)]
HOMOGRAPHIES = [f"H_1_{i}" for i in range(2, 7)]

# high-resolution sequences that D2-Net and later works exclude (108-scene variant)
EXCLUDED_108 = ["i_contruction", "i_crownnight", "i_dc", "i_pencils", "i_whitebuilding",
                "v_artisans", "v_astronautis", "v_talent"]


def check_homography(path):
    """Return (H, problem). problem is None if the file is a valid 3x3 homography."""
    try:
        H = np.loadtxt(path)
    except Exception as e:
        return None, f"cannot parse ({e})"
    if H.shape != (3, 3):
        return None, f"shape {H.shape}, expected (3, 3)"
    if not np.all(np.isfinite(H)):
        return None, "contains NaN/inf"
    if abs(np.linalg.det(H)) < 1e-12:
        return None, "singular matrix"
    return H, None


def main():
    parser = argparse.ArgumentParser(description="Check HPatches sequences structure")
    parser.add_argument("root", help="path to hpatches-sequences-release")
    parser.add_argument("--read-images", action="store_true",
                        help="also open every image with OpenCV and collect sizes (slower)")
    args = parser.parse_args()

    if not os.path.isdir(args.root):
        print(f"[FAIL] directory not found: {args.root}")
        sys.exit(1)

    entries = sorted(os.listdir(args.root))
    seqs = [d for d in entries if os.path.isdir(os.path.join(args.root, d))]
    other = [d for d in entries if d not in seqs]
    seqs_i = [s for s in seqs if s.startswith("i_")]
    seqs_v = [s for s in seqs if s.startswith("v_")]
    unknown = [s for s in seqs if s not in seqs_i + seqs_v]

    problems = []
    if args.read_images:
        import cv2
        sizes = {}

    n_pairs = 0
    for seq in seqs:
        d = os.path.join(args.root, seq)
        files = set(os.listdir(d))

        missing = [f for f in IMAGES + HOMOGRAPHIES if f not in files]
        if missing:
            problems.append(f"{seq}: missing {missing}")

        for h in HOMOGRAPHIES:
            if h in files:
                _, err = check_homography(os.path.join(d, h))
                if err:
                    problems.append(f"{seq}/{h}: {err}")

        if args.read_images:
            for im in IMAGES:
                if im in files:
                    img = cv2.imread(os.path.join(d, im))
                    if img is None:
                        problems.append(f"{seq}/{im}: cannot read image")
                    else:
                        sizes[(seq, im)] = img.shape[:2]

        n_pairs += sum(1 for i in range(2, 7) if f"{i}.ppm" in files and f"H_1_{i}" in files
                       and "1.ppm" in files)

    def status(ok):
        return "OK  " if ok else "FAIL"

    print(f"root: {os.path.abspath(args.root)}\n")
    print(f"[{status(len(seqs) == EXPECTED_SEQ)}] sequences: {len(seqs)} (expected {EXPECTED_SEQ})")
    print(f"[{status(len(seqs_i) == EXPECTED_I)}] illumination i_*: {len(seqs_i)} (expected {EXPECTED_I})")
    print(f"[{status(len(seqs_v) == EXPECTED_V)}] viewpoint v_*: {len(seqs_v)} (expected {EXPECTED_V})")
    print(f"[{status(n_pairs == 5 * EXPECTED_SEQ)}] complete pairs: {n_pairs} (expected {5 * EXPECTED_SEQ})")
    print(f"[{status(not unknown)}] folders without i_/v_ prefix: {unknown if unknown else 'none'}")
    if other:
        print(f"[note] non-folder entries in root (ignored): {other}")

    present_excl = [s for s in EXCLUDED_108 if s in seqs]
    print(f"[note] high-res sequences for the 108-scene variant present: "
          f"{len(present_excl)}/{len(EXCLUDED_108)}")

    if args.read_images and sizes:
        hw = np.array(list(sizes.values()))
        print(f"\nimages read: {len(sizes)}")
        print(f"  height: min {hw[:, 0].min()}, max {hw[:, 0].max()}, median {int(np.median(hw[:, 0]))}")
        print(f"  width:  min {hw[:, 1].min()}, max {hw[:, 1].max()}, median {int(np.median(hw[:, 1]))}")
        largest = sorted(sizes.items(), key=lambda kv: -kv[1][0] * kv[1][1])[:3]
        print("  largest: " + ", ".join(f"{s}/{im} {h}x{w}" for (s, im), (h, w) in largest))

    if problems:
        print(f"\n[FAIL] {len(problems)} problem(s):")
        for p in problems[:50]:
            print("  " + p)
        if len(problems) > 50:
            print(f"  ... and {len(problems) - 50} more")

    ok = (not problems and not unknown and len(seqs) == EXPECTED_SEQ
          and len(seqs_i) == EXPECTED_I and len(seqs_v) == EXPECTED_V
          and n_pairs == 5 * EXPECTED_SEQ)
    print("\nRESULT: " + ("dataset structure is OK" if ok else "dataset structure has problems"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
