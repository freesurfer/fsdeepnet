#!/usr/bin/env python3

import os
import sys
import argparse

ref = '''
If you use SynthStrip in your analysis, please cite:
----------------------------------------------------
SynthStrip: Skull-Stripping for Any Brain Image
A Hoopes, JS Mora, AV Dalca, B Fischl, M Hoffmann
NeuroImage 206 (2022), 119474
https://doi.org/10.1016/j.neuroimage.2022.119474

Website: https://synthstrip.io
'''

# parse command line
p = argparse.ArgumentParser(description='Robust, universal skull-stripping for brain images of any type.')
p.add_argument('-i', '--image', metavar='FILE', required=True, help='input image to skullstrip')
p.add_argument('-o', '--out', metavar='FILE', help='save stripped image to file')
p.add_argument('-m', '--mask', metavar='FILE', help='save binary brain mask to file')
p.add_argument('-d', '--sdt', metavar='FILE', help='save distance transform to file')
p.add_argument('-g', '--gpu', action='store_true', help='use the GPU')
p.add_argument('-b', '--border', default=1, type=float, help='mask border threshold in mm, defaults to 1')
p.add_argument('-t', '--threads', type=int, help='PyTorch CPU threads, PyTorch default if unset')
p.add_argument('-f', '--fill', type=float, help='BG fill value, defaults to min(image.min, 0)')
p.add_argument('--no-csf', action='store_true', help='exclude CSF from brain border')
p.add_argument('--model', metavar='FILE', help='alternative model weights')
p.add_argument('-v', '--version', action="store_true", help='print SynthStrip version')

if len(sys.argv) == 1 or '-h' in sys.argv or '--help' in sys.argv:
    p.print_help()
    print(ref)
    exit(1)
if ("-v" in sys.argv or "--version" in sys.argv):
    v = os.environ.get('SYNTHSTRIP_VERSION', 'unknown')
    print(v)
    exit(1)

args = p.parse_args()

# do not wait for third-party imports just to show usage
import torch
import torch.nn as nn
import numpy as np
import surfa as sf

# sanity check on the inputs
if not args.out and not args.mask and not args.sdt:
    sf.system.fatal('Must provide at least one -o, -m, or -d output flag.')

# necessary for speed gains (I think)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

# configure device
if args.gpu:
    if not torch.cuda.is_available():
        sf.system.fatal('-g flag provided but CUDA is not available')
    device = torch.device('cuda')
    device_name = 'GPU'
else:
    device = torch.device('cpu')
    device_name = 'CPU'

if args.threads is not None:
    torch.set_num_threads(args.threads)

def extend_sdt(sdt, border=1):
    """Extend SynthStrip's narrow-band signed distance transform (SDT).

    Recompute the positive outer part of the SDT estimated by SynthStrip, for
    borders that likely exceed the 4-5 mm band. Keeps the negative inner part
    intact and only computes the outer part where needed to save time.

    Parameters
    ----------
    sdt : sf.Volume
        Narrow-band signed distance transform estimated by SynthStrip.
    border : float, optional
        Mask border threshold in millimeters.

    Returns
    -------
    sdt : sf.Volume
        Extended SDT.

    """
    if border < int(sdt.max()):
        return sdt

    # Find bounding box.
    mask = sdt < 1
    keep = np.nonzero(mask)
    low = np.min(keep, axis=-1)
    upp = np.max(keep, axis=-1)

    # Add requested border.
    gap = int(border + 0.5)
    low = (max(i - gap, 0) for i in low)
    upp = (min(i + gap, d - 1) for i, d in zip(upp, mask.shape))

    # Compute EDT within bounding box. Keep interior values.
    ind = tuple(slice(a, b + 1) for a, b in zip(low, upp))
    out = np.full_like(sdt, fill_value=100)
    out[ind] = sf.Volume(mask[ind]).distance()
    out[keep] = sdt[keep]

    return sdt.new(out)

# configure model
print(f'Configuring model on the {device_name}')

# load model weights
if args.model is not None:
    modelfile = args.model
    print('Using custom model weights')
else:
    version = '1'
    print(f'Running SynthStrip model version {version}')
    fshome = os.environ.get('FREESURFER_HOME')
    if fshome is None:
        sf.system.fatal('FREESURFER_HOME env variable must be set! Make sure FreeSurfer is properly sourced.')
    if args.no_csf:
        print('Excluding CSF from brain boundary')
        modelfile = f'synthstrip.nocsf.{version}.pth'
    else:
        modelfile = f'synthstrip.{version}.pth'

from fsdeepnet.utils import utility as utils
model = utils.load_pretrained(modelfile, device)

# load input volume
image = sf.load_volume(args.image)
print(f'Input image read from: {args.image}')

# loop over frames (try not to keep too much data in memory)
print(f'Processing frame (of {image.nframes}):', end=' ', flush=True)
dist = []
mask = []
for f in range(image.nframes):
    print(f + 1, end=' ', flush=True)
    frame = image.new(image.framed_data[..., f])

    # conform, fit to shape with factors of 64
    conformed = frame.conform(voxsize=1.0, dtype='float32', method='nearest', orientation='LIA')
    conformed = conformed.crop_to_bbox()
    target_shape = np.clip(np.ceil(np.array(conformed.shape[:3]) / 64).astype(int) * 64, 192, 320)
    conformed = conformed.reshape(target_shape)

    # normalize
    conformed -= conformed.min()
    conformed = (conformed / conformed.percentile(99)).clip(0, 1)
    inp = torch.from_numpy(conformed.data[np.newaxis, np.newaxis]).to(device)

    # predict the sdt
    with torch.no_grad():
        sdt = model(inp).squeeze().cpu()

    # extend the sdt if needed, unconform
    sdt = extend_sdt(conformed.new(sdt), border=args.border)
    sdt = sdt.resample_like(image, fill=100)
    dist.append(sdt)

    # extract mask, find largest CC to be safe
    mask.append((sdt < args.border).connected_component_mask(k=1, fill=True))

# combine frames and end line
dist = sf.stack(dist)
mask = sf.stack(mask)
print('done')

# write the masked output
if args.out:
    fill = np.min([image.min(), 0]) if args.fill is None else args.fill
    image[mask == 0] = fill
    image.save(args.out)
    print(f'Set background to: {fill}')
    print(f'Masked image saved to: {args.out}')

# write the brain mask
if args.mask:
    image.new(mask).save(args.mask)
    print(f'Binary brain mask saved to: {args.mask}')

# write the distance transform
if args.sdt:
    image.new(dist).save(args.sdt)
    print(f'Distance transform saved to: {args.sdt}')

print(ref)
