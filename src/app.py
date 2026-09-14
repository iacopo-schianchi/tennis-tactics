import argparse
from processor import VideoProcessor

FPS = 30

def start_pass_type(value):
    if value == "full":
        return "full"
    return int(value)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="assets/clip.mp4")
    parser.add_argument("--context", default=None, help="Path to an existing context JSON file to load")
    parser.add_argument(
        "--start-pass",
        default=0,
        type=start_pass_type,
        help="Pass index to start from, or 'full' to skip all passes and use the loaded context as-is"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    vp = VideoProcessor('p1', 'p2', FPS)
    vp.process(args.video, context_path=args.context, start_pass=args.start_pass)