import numpy as np

class CentroidTracker:
    """Greedy nearest-centroid tracker; basic baseline, not occlusion-proof."""
    def __init__(self, max_distance=35.0, max_missed=5):
        self.max_distance = float(max_distance)
        self.max_missed = int(max_missed)
        self.next_id = 1
        self.tracks = {}

    def update(self, detections):
        points = [tuple(map(float,d["centroid"])) for d in detections]
        pairs = []
        for tid, data in self.tracks.items():
            for i, point in enumerate(points):
                d = float(np.linalg.norm(np.asarray(data["centroid"])-np.asarray(point)))
                if d <= self.max_distance:
                    pairs.append((d,tid,i))
        pairs.sort()
        used_t, used_d = set(), set()
        for _,tid,i in pairs:
            if tid not in used_t and i not in used_d:
                self.tracks[tid] = {"centroid":points[i],"missed":0}
                used_t.add(tid); used_d.add(i)
        for i,point in enumerate(points):
            if i not in used_d:
                tid = self.next_id; self.next_id += 1
                self.tracks[tid] = {"centroid":point,"missed":0}
                used_t.add(tid)
        for tid in list(self.tracks):
            if tid not in used_t:
                self.tracks[tid]["missed"] += 1
                if self.tracks[tid]["missed"] > self.max_missed:
                    del self.tracks[tid]
        return {tid:d["centroid"] for tid,d in self.tracks.items() if d["missed"] == 0}
