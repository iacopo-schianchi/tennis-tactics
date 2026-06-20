import cv2
from modules.court import CourtDetector
from orchestrator import MatchOrchestrator

class VideoProcessor:
    def __init__(self):
        self.modules = [
            CourtDetector(),
        ]
        self.orchestrator = MatchOrchestrator()
    
    def process(self, video_path):
        cap = cv2.VideoCapture(video_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_id = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_data = self._run_perception(frame, frame_id)
            self.orchestrator.process_frame(frame_data)

            if frame_id % 100 == 0:
                print(f"Processed {frame_id}/{total_frames} frames")
            
            frame_id += 1
        
        cap.release()

    def _run_perception(self, frame, frame_id):
        frame_data = {}

        for module in self.modules:
            frame_data.update(module.process(frame, frame_id))
        
        return frame_data