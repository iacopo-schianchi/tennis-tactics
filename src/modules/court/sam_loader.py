import numpy as np

from segment_anything import SamPredictor, sam_model_registry
from pathlib import Path

CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "models"
    / "sam_vit_b_01ec64.pth"
)

def load_sam_model(device=None):
    sam = sam_model_registry["vit_b"](checkpoint=CHECKPOINT_PATH).to(device)
    predictor = SamPredictor(sam)
    
    return predictor

# return best sam mask
def run_sam_segmentation(predictor, raw_image):
    image_np = np.array(raw_image)
    height, width = image_np.shape[:2]

    predictor.set_image(image_np)

    input_point = np.array([[width // 2, height // 2]])
    input_label = np.array([1])
    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True
    )

    # get biggest mask
    best_mask = masks[masks.sum(axis=(1, 2)).argmax()]
    return best_mask