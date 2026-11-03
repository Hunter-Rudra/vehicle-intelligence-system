# Vehicle Intelligence System (v0.1)

From-scratch computer vision using NumPy as the only third-party dependency.

## Scope
Grayscale conversion, NumPy Gaussian filtering, running-average background subtraction, connected components, centroid tracking, and virtual-line crossing counts.

This version processes NumPy arrays / `.npy` frame sequences. It does not decode MP4 files and does not classify cars, bikes, or scooters. Motion blobs are not semantic vehicle detections.

## Install and run
```bash
python -m venv .venv
.venv\Scripts\activate
pip install numpy
python -m vehicle_intelligence.demo
python -m unittest discover -s tests -v
```

## Own frame sequence
Save frames as `(N,H,W,3)` RGB or `(N,H,W)` grayscale in `frames.npy`:
```python
import numpy as np
from vehicle_intelligence.pipeline import VehiclePipeline
frames = np.load("frames.npy")
pipeline = VehiclePipeline()
for frame in frames:
    result = pipeline.process(frame)
    print(len(result["detections"]), result["crossings"])
```

## Limitations
Fixed-camera assumption; shadows and lighting changes cause false positives; simple centroid matching can fail under occlusion. The demo uses synthetic rectangles, not real CCTV footage.
