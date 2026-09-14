import argparse
from processor import VideoProcessor
from uuid import UUID, uuid4

FPS = 30

def start_pass_type(value):
    if value == "full":
        return "full"
    return int(value)

def uuid_type(value):
    try:
        return str(UUID(value))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f'invalid UUID: {value}') from error

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
    parser.add_argument(
        "--persist",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Persist processed video to Supabase",
    )
    parser.add_argument("--far-id", type=uuid_type, default=None, help="UUID of far player")
    parser.add_argument("--near-id", type=uuid_type, default=None, help="UUID of near player")
    parser.add_argument("--far-name", default="Far Player", help="Name of far player")
    parser.add_argument("--near-name", default="Near Player", help="Name of near player")
    parser.add_argument(
        "--far-handedness",
        choices=("right", "left"),
        default=None,
        help="Handedness of far player",
    )
    parser.add_argument(
        "--near-handedness",
        choices=("right", "left"),
        default=None,
        help="Handedness of near player",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    far_id = args.far_id or str(uuid4())
    near_id = args.near_id or str(uuid4())

    far_player = {
        "id": far_id,
        "name": args.far_name,
        "handedness": args.far_handedness,
    }
    near_player = {
        "id": near_id,
        "name": args.near_name,
        "handedness": args.near_handedness,
    }

    context = None if args.context == 'None' else args.context

    vp = VideoProcessor(far_player, near_player, FPS)
    vp.process(args.video, context_path=context, start_pass=args.start_pass, persist=args.persist)