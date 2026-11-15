"""Evaluate saved saliency masks; no model implementation is required."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from evaluation.saliency_metric import cal_em, cal_fm, cal_mae, cal_sm, cal_wfm


def png_files(directory):
    files = [p for p in directory.iterdir() if p.is_file()]
    other_images = {'.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp', '.npy'}
    if any(p.suffix.lower() in other_images for p in files):
        raise ValueError('This evaluator accepts PNG masks only: ' + str(directory))
    return {p.name: p for p in files if p.suffix.lower() == '.png'}


def load_mask(path, size=None, ground_truth=False):
    with Image.open(path) as image:
        if image.mode not in ('1', 'L'):
            raise ValueError('Expected a single-channel 8-bit or binary PNG mask: ' + str(path))
        image = image.convert('L')
        if size is not None:
            image = image.resize(size, Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float64)
    if ground_truth:
        return (array > (0 if array.max() <= 1 else 127.5)).astype(np.float64)
    return array / 255.0


def paired_files(pred_root, gt_root, gt_subdir):
    pred_root, gt_root = Path(pred_root), Path(gt_root)
    if not pred_root.is_dir() or not gt_root.is_dir():
        raise FileNotFoundError('Prediction and ground-truth roots must both exist.')
    if gt_subdir and (len(Path(gt_subdir).parts) != 1 or Path(gt_subdir).drive
                      or '/' in gt_subdir or '\\' in gt_subdir or gt_subdir in ('.', '..')):
        raise ValueError('--gt-subdir must be one directory name, or an empty string.')
    videos = sorted(p.name for p in gt_root.iterdir() if p.is_dir())
    predicted = sorted(p.name for p in pred_root.iterdir() if p.is_dir())
    if not videos or videos != predicted:
        raise ValueError('Prediction and GT video directories must match exactly.')
    pairs = []
    for video in videos:
        target = gt_root/video/gt_subdir if gt_subdir else gt_root/video
        if not target.is_dir():
            raise FileNotFoundError('Missing GT directory: ' + str(target))
        gt, pred = png_files(target), png_files(pred_root/video)
        if not gt or set(gt) != set(pred):
            missing, extra = sorted(set(gt)-set(pred)), sorted(set(pred)-set(gt))
            raise ValueError('Frame mismatch in {}: {} missing, {} extra PNG predictions.'.format(video, len(missing), len(extra)))
        pairs.extend((video, name, pred[name], gt[name]) for name in sorted(gt))
    return pairs


def evaluate(pred_root, gt_root, gt_subdir='gt', resize=False):
    pairs = paired_files(pred_root, gt_root, gt_subdir)
    sm, em, fm, mae, wfm = cal_sm(), cal_em(), cal_fm(), cal_mae(), cal_wfm()
    adaptive_f = []
    empty = 0
    per_video = {}
    for video, name, pred_path, gt_path in pairs:
        target = load_mask(gt_path, ground_truth=True)
        if min(target.shape) < 2:
            raise ValueError('Masks must have at least two rows and columns: ' + str(gt_path))
        pred = load_mask(pred_path)
        if pred.shape != target.shape:
            if not resize:
                raise ValueError('Mask sizes differ for {}/{}. Use --resize-predictions explicitly.'.format(video, name))
            pred = load_mask(pred_path, size=(target.shape[1], target.shape[0]))
        sm.update(pred, target)
        em.update(pred, target)
        fm.update(pred, target)
        mae.update(pred, target)
        wfm.update(pred, target)
        # Unlike the legacy PR-curve accumulator, retain empty GT in adaptive F.
        threshold = min(2*float(pred.mean()), 1.0)
        selected, foreground = pred >= threshold, target > .5
        tp = int(np.logical_and(selected, foreground).sum())
        if tp:
            precision = tp/int(selected.sum())
            recall = tp/int(foreground.sum())
            adaptive_f.append(1.3*precision*recall/(.3*precision+recall))
        else:
            adaptive_f.append(0.0)
        empty += int(not foreground.any())
        per_video[video] = per_video.get(video, 0)+1
    max_f = float(fm.show()[0]) if fm.precision else None
    return {'frames': len(pairs), 'videos': len(per_video), 'empty_gt_frames': empty,
            'S_m': float(sm.show()), 'E_adaptive': float(em.show()),
            'F_adaptive': float(np.mean(adaptive_f)), 'F_max': max_f,
            'F_weighted': float(wfm.show()), 'MAE': float(mae.show()),
            'frames_per_video': per_video,
            'evaluation': {'input': 'Single-channel 8-bit or binary PNG masks.',
                'aggregation': 'All-frame mean for S/E/adaptive F/weighted F/MAE; frame-weighted across videos.',
                'adaptive_threshold': 'min(2*mean(prediction),1), beta squared=0.3.',
                'F_max': 'Maximum F of dataset-averaged PR curves, following the included legacy utility; empty GT excluded.',
                'empty_gt': 'Adaptive F and weighted F are zero; MAE/S/E follow the included definitions. F_max is null when all GT masks are empty.',
                'resize_predictions': bool(resize),
                'scope': 'Offline mask evaluation only; not a full model reproduction protocol.'}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pred-root', type=Path, required=True)
    parser.add_argument('--gt-root', type=Path, required=True)
    parser.add_argument('--gt-subdir', default='gt', help="Use '' if masks are directly inside each GT video directory.")
    parser.add_argument('--resize-predictions', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        raise FileExistsError('Choose a new output file: ' + str(args.output))
    result = evaluate(args.pred_root, args.gt_root, args.gt_subdir, args.resize_predictions)
    text = json.dumps(result, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(text+'\n')
    print(text)


if __name__ == '__main__':
    main()
