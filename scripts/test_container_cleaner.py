"""Test script for container_cleaner and M3 removal strategies."""
import os
import sys
import numpy as np
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from watermark_remover.container_cleaner import fill_masked_area_rgb, telea_inpaint_masked_area
from watermark_remover import doc_segment, doc_core, ui

def test_container_isolation():
    print("1. Testing container boundary color isolation...")
    # Create image 100x100 with two zones separated by a dark vertical line at x=50
    # Left zone: white (250, 250, 250)
    # Right zone: shaded/cream (210, 210, 210)
    # Vertical line at x=50: dark rule (40, 40, 40)
    test_img = np.ones((100, 100, 3), dtype=np.uint8) * 250
    test_img[:, 51:] = 210
    test_img[:, 50] = 40  # Border

    # Place a mask in left zone: (y: 30-70, x: 20-35)
    mask = np.zeros((100, 100), dtype=bool)
    mask[30:70, 20:35] = True
    test_img[mask] = 185  # Faint watermark

    filled = fill_masked_area_rgb(test_img, mask)

    # Verify masked region filled with left zone color (250), not contaminated by right zone (210)
    filled_patch = filled[30:70, 20:35]
    mean_val = np.mean(filled_patch)
    print(f"   Left zone filled mean: {mean_val:.1f} (expected ~250, border stopped 210)")
    assert mean_val >= 245, f"Expected left zone >= 245, got {mean_val}"

    # Verify border at x=50 was preserved
    assert np.all(filled[:, 50] == 40), "Border at x=50 was altered!"
    # Verify right zone was completely untouched
    assert np.all(filled[:, 51:] == 210), "Right zone was altered!"
    print("   [PASS] Container boundary isolation verified!")

def test_m3_strategies_synthetic():
    print("2. Testing all 4 M3 removal strategies on synthetic input...")
    dummy = np.ones((200, 200, 3), dtype=np.uint8) * 240
    # Add dark text
    dummy[80:120, 80:120] = 30

    strategies = [
        doc_segment.STRATEGY_TEMPLATE,
        doc_segment.STRATEGY_CONTAINER_FILL,
        doc_segment.STRATEGY_SUBTRACTIVE,
        doc_segment.STRATEGY_TELEA,
    ]

    for strat in strategies:
        cleaned, status = doc_segment.clean_document_segment(
            dummy,
            conf=0.25,
            model_choice="Finetuned (AriaTender)",
            removal_strategy=strat,
        )
        assert cleaned.shape == dummy.shape
        assert isinstance(status, dict)
        print(f"   [PASS] Strategy '{strat}' ran: {status['removal_strategy']}")

def test_real_sample():
    sample_path = os.path.join(REPO_ROOT, "sample images", "sample_watermarked.jpg")
    if not os.path.exists(sample_path):
        print("Sample image not found, skipping real sample test.")
        return

    print("3. Testing Container Fill on real sample image:", sample_path)
    img_pil = Image.open(sample_path).convert("RGB")
    img_np = np.array(img_pil)

    cleaned_np, status = doc_segment.clean_document_segment(
        img_np,
        conf=0.25,
        model_choice="Finetuned (Half-Frozen)",
        removal_strategy=doc_segment.STRATEGY_CONTAINER_FILL,
    )
    print(f"   Instances: {status['instances_found']}, Accepted: {status['instances_accepted']}")
    print(f"   Status message: {status['message']}")
    assert cleaned_np.shape == img_np.shape
    print("   [PASS] Real sample processed cleanly!")

def test_doc_core_and_ui():
    print("4. Testing doc_core.clean_document integration...")
    dummy = Image.fromarray(np.ones((100, 100, 3), dtype=np.uint8) * 240)
    cleaned_img, status_str = doc_core.clean_document(
        dummy,
        method=doc_core.METHOD_SEGMENT,
        seg_strategy=doc_segment.STRATEGY_CONTAINER_FILL,
        save_dataset=False,
    )
    assert cleaned_img is not None
    print(f"   doc_core result: {status_str[:60]}...")

    print("5. Testing ui.build_ui()...")
    demo = ui.build_ui()
    assert demo is not None
    print("   [PASS] UI built successfully!")

if __name__ == "__main__":
    test_container_isolation()
    test_m3_strategies_synthetic()
    test_real_sample()
    test_doc_core_and_ui()
    print("\nALL CONTAINER CLEANER TESTS PASSED SUCCESSFULLY!")
