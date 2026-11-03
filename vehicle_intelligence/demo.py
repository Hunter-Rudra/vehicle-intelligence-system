from .pipeline import VehiclePipeline
from .synthetic import moving_rectangle_sequence

def main():
    pipeline = VehiclePipeline(min_area=12, motion_threshold=15, line_y=65)
    for i, frame in enumerate(moving_rectangle_sequence()):
        result = pipeline.process(frame)
        if result["detections"]:
            print(f"frame={i:02d} blobs={len(result['detections'])} "
                  f"track_ids={list(result['tracks'])} crossings={result['crossings']}")
    print("Demo complete: motion blobs only, not vehicle classification.")

if __name__ == "__main__":
    main()
