import os
import cv2
import numpy as np

def create_illustrative_test_clip(output_path: str = "data/illustrative_traffic_sample.mp4", num_frames: int = 120):
    """
    Creates a small illustrative video clip labeled clearly as an illustrative test fixture.
    Shows multiple vehicle tracks with realistic dimensions and trajectories:
    - Track 1: Car moving downward (legal).
    - Track 2: Motorcycle moving upward against traffic (wrong-side driving).
    - Track 3: Vehicle crossing stop line while signal is RED (red-light violation).
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    width, height = 640, 480
    fps = 25.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Road background
    base_frame = np.full((height, width, 3), (70, 70, 70), dtype=np.uint8)
    # Road lanes (asphalt dark gray, white lane dividers)
    cv2.rectangle(base_frame, (100, 0), (540, height), (50, 50, 50), -1)
    # Dashed center line
    for y in range(0, height, 40):
        cv2.line(base_frame, (320, y), (320, y + 20), (240, 240, 240), 2)

    # Road boundaries
    cv2.line(base_frame, (100, 0), (100, height), (255, 255, 255), 3)
    cv2.line(base_frame, (540, 0), (540, height), (255, 255, 255), 3)

    for i in range(num_frames):
        frame = base_frame.copy()

        # Banner marking this footage clearly as illustrative test data
        cv2.putText(
            frame,
            "ILLUSTRATIVE TEST DATA - SMALL ILLUSTRATIVE BENCHMARK",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA
        )

        # Vehicle 1: Car moving downward along right lane (Legal direction)
        car_y = int(50 + i * 3.5) % height
        cv2.rectangle(frame, (380, car_y), (460, car_y + 90), (0, 140, 255), -1)  # Orange car
        cv2.putText(frame, "Car", (390, car_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Vehicle 2: Motorcycle moving upward along left lane (Wrong-side driving against allowed [0, 1])
        bike_y = int(height - 60 - i * 3.2)
        if bike_y > -60:
            cv2.rectangle(frame, (180, bike_y), (220, bike_y + 55), (200, 50, 50), -1)  # Blue bike
            cv2.putText(frame, "Moto", (182, bike_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        # Vehicle 3: Crossing stop line at frame 50-90
        if 40 <= i <= 95:
            red_viol_y = int(320 + (i - 40) * 3.0)
            cv2.rectangle(frame, (250, red_viol_y), (330, red_viol_y + 70), (50, 180, 50), -1)
            cv2.putText(frame, "Van", (260, red_viol_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        out.write(frame)

    out.release()
    print(f"Sample illustrative test clip generated at: {output_path} ({num_frames} frames, {width}x{height})")

if __name__ == "__main__":
    create_illustrative_test_clip()
