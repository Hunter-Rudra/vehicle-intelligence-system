class LineCounter:
    def __init__(self, y, deadband=2.0):
        self.y = float(y)
        self.deadband = float(deadband)
        self.previous_side = {}
        self.counted_ids = set()
        self.count = 0

    def update(self, tracks):
        for tid, (_,cy) in tracks.items():
            side = -1 if cy < self.y-self.deadband else (1 if cy > self.y+self.deadband else 0)
            if side == 0:
                continue
            old = self.previous_side.get(tid)
            if old is not None and old != side and tid not in self.counted_ids:
                self.count += 1
                self.counted_ids.add(tid)
            self.previous_side[tid] = side
        return self.count
