# Dataset Template

Folder layout:
```
images/train
images/val
labels/train
labels/val
```

- Label tip as small bounding boxes (class 0 = stick_tip).
- 300–800 frames; include various lighting, angles, tempos.
- Recommend augmentations: motion blur, brightness/contrast, affine, cutout.
