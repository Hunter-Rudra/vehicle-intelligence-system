from collections import deque
import numpy as np

def connected_components(mask, min_area=20):
    """8-connected component extraction; boxes are (x,y,width,height)."""
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("mask must be 2D")
    h, w = binary.shape
    seen = np.zeros_like(binary)
    result = []
    for y in range(h):
        for x in range(w):
            if not binary[y,x] or seen[y,x]:
                continue
            q = deque([(y,x)])
            seen[y,x] = True
            xs, ys = [], []
            while q:
                cy,cx = q.popleft()
                xs.append(cx); ys.append(cy)
                for dy in (-1,0,1):
                    for dx in (-1,0,1):
                        ny,nx = cy+dy,cx+dx
                        if (dx or dy) and 0 <= ny < h and 0 <= nx < w and binary[ny,nx] and not seen[ny,nx]:
                            seen[ny,nx] = True
                            q.append((ny,nx))
            area = len(xs)
            if area >= min_area:
                x0,x1,y0,y1 = min(xs),max(xs),min(ys),max(ys)
                result.append({"bbox":(x0,y0,x1-x0+1,y1-y0+1),
                               "area":area,
                               "centroid":((x0+x1)/2.0,(y0+y1)/2.0)})
    return result
