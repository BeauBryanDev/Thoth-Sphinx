#!/usr/bin/env python3

# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np
import argparse
from pathlib import Path

# Settings for the pipeline

PRESETS = {
    "default": {
        "clahe_clip":      2.0,
        "clahe_grid":      8,
        "denoise_h":       6,  
        "denoise_method":  "nlm",  # "gaussian" | "nlm" 
        "unsharp_amount":  0.6,    
        "unsharp_radius":  1.5,   
        "gamma":           1.05,   
    },
    "aggressive": {
        "clahe_clip":      3.5,
        "clahe_grid":      8,
        "denoise_h":       10,
        "denoise_method":  "nlm",
        "unsharp_amount":  1.0,
        "unsharp_radius":  2.0,
        "gamma":           0.90,  
    },
    "gentle": {
        "clahe_clip":      1.5,
        "clahe_grid":      16,     
        "denoise_h":       3,
        "denoise_method":  "nlm",
        "unsharp_amount":  0.3,
        "unsharp_radius":  1.0,
        "gamma":           1.0,
    },
    "fast": {
        "clahe_clip":      2.0,
        "clahe_grid":      8,
        "denoise_h":       15,
        "denoise_method":  "bilateral",
        "unsharp_amount":  0.6,
        "unsharp_radius":  1.5,
        "gamma":           1.05,
    }
}


# Pipeline steps
# Constrast Adaptive Histogram Equalization (CLAHE) on L channel of LAB space
def apply_clahe(img: np.ndarray,
                clip: float = 2.0,
                grid: int = 8) -> np.ndarray:
    """
    CLAHE on L channel of LAB space.
    """
    # My YOLO model was trained on BGR images, so convert to LAB space ( Rock Color from Ancient Temples )
    lab      = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b  = cv2.split(lab)
    # Convert to LAB, apply CLAHE on L channel, then convert back to BGR
    clahe    = cv2.createCLAHE(clipLimit=clip,
                                tileGridSize=(grid, grid))
    l_clahe  = clahe.apply(l)
    merged   = cv2.merge([l_clahe, a, b])
    
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def apply_denoising(img: np.ndarray,
                    h: float = 6,
                    method: str = "nlm" # gaussian | nlm 
                    ) -> np.ndarray:
    """
    Non-local means denoising — keep borders better than GaussianBlur.
    h: strength of the filter. >10 erases fine details of the sign.
    """
    if h <= 0:
        return img
    
    if method == "nlm":
        # Non-local means denoising — keep borders better than GaussianBlur.
        return cv2.fastNlMeansDenoisingColored(
            img,
            None,
            h=h,
            hColor=h,
            templateWindowSize=7, # search window size for similar patches px 
            searchWindowSize=21
        )
        
        
    elif method == "gaussian":
        # Gaussian denoising — keep borders better than bilateral filter.
        return cv2.fastNlMeansDenoisingColored(
            img,
            None,
            h=h,
            hColor=h,
            templateWindowSize=7,
            searchWindowSize=21,
            sigmaColor=0,
            sigmaSpace=0
        )
        
    elif method == "bilateral":
        # Bilateral denoising — keep borders better than GaussianBlur.
        return cv2.bilateralFilter(img, d=9, sigmaColor=h, sigmaSpace=h)
    
    else:
        raise ValueError(f"Invalid denoising method: {method}")


def apply_unsharp_mask(img: np.ndarray,  # input image
                        amount: float = 0.6,  # 0.0 = no effect, 1.5 = very aggressive
                        radius: float = 1.5 # use a larger radius for more aggressive effect
                        ) -> np.ndarray:
    """
    Unsharp mask — highlights the edges of the carving in stone.
    amount: 0.0 = no effect, 1.5 = very aggressive.
    Equation: sharpened = original + amount * (original - blurred)
    """
    if amount <= 0:
        return img
    
    blurred   = cv2.GaussianBlur(img, (0, 0), radius)
    sharpened = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
    
    return sharpened


def apply_gamma(img: np.ndarray, gamma: float = 1.05) -> np.ndarray:
    """
    Gamma correction by lookup table — fast and without artifacts.
    gamma < 1.0 → brightens (useful for dark photos of tombs)
    gamma > 1.0 → darkens slightly (increases visual contrast)
    gamma = 1.0 → no change
    """
    if gamma == 1.0:
        return img
    
    inv_gamma = 1.0 / gamma
    
    table     = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in range(256)
    ], dtype=np.uint8)
    
    return cv2.LUT(img, table)


# Main pipeline

def enhance(img: np.ndarray,
            preset: str = "default",
            custom: dict | None = None) -> np.ndarray:
    """
    Complete image enhancement pipeline.

    Args:
        img:    imagen BGR (numpy array) — output de cv2.imread()
        preset: 'default' | 'aggressive' | 'gentle' | 'fast'
        custom: dict with custom parameters (overrides preset)

    Returns:
        enhanced BGR image (same size as input)
    """
    if preset not in PRESETS:
        raise ValueError(
            f"Unknown preset: {preset!r}. "
            f"Choose from {sorted(PRESETS.keys())}."
        )
    
    cfg = PRESETS.get(preset, PRESETS["default"]).copy()
    
    if custom:
        cfg.update(custom)

    # Step 1 — CLAHE
    out = apply_clahe(img,
                      clip=cfg["clahe_clip"],
                      grid=cfg["clahe_grid"])

    # Step 2 — Denoising
    out = apply_denoising(out, 
                          h=cfg["denoise_h"],
                          method=cfg["denoise_method"])

    # Step 3 — Unsharp mask
    out = apply_unsharp_mask(out,
                              amount=cfg["unsharp_amount"],
                              radius=cfg["unsharp_radius"])

    # Step 4 — Gamma
    out = apply_gamma(out, gamma=cfg["gamma"])

    return out


def compare_side_by_side(original: np.ndarray,
                          enhanced: np.ndarray,
                          label: str = "") -> np.ndarray:
    """Generate side-by-side comparison image original | enhanced for inspection."""
    h = max(original.shape[0], enhanced.shape[0])

    def pad_h(img, target_h):
        ph = target_h - img.shape[0]
        if ph > 0:
            return np.vstack([img, np.zeros((ph, img.shape[1], 3),
                                            dtype=np.uint8)])
        return img

    orig_pad = pad_h(original, h)
    enh_pad  = pad_h(enhanced, h)

    divider  = np.full((h, 3, 3), 100, dtype=np.uint8)
    combined = np.hstack([orig_pad, divider, enh_pad])

    # Labels
    cv2.putText(combined, "ORIGINAL", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(combined, f"ENHANCED ({label})",
                (orig_pad.shape[1] + 13, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 220, 100), 1)
    return combined

# This script file will turn out to be app/utils/enhance_img.py 
# when this bakcend app is built in FastAPI
# for now it is a standalone script to test the enhancement pipeline
# It will become the dritical step 0 . after user upload an image from frontend
# and before the image is sent to the model for inference 

# Entry point

def main():
    parser = argparse.ArgumentParser(
        description="SphinxEyes — Image preprocessor for inference"
    )
    parser.add_argument("input", help="Input image")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: <name>_enhanced.jpg)")
    parser.add_argument("--preset", type=str, default="default",
                        choices=["default", "aggressive", "gentle", "fast"],
                        help="Preset de mejora (default: default)")
    parser.add_argument("--show", action="store_true",
                        help="Show original vs enhanced comparison")
    parser.add_argument("--compare-all", action="store_true",
                        help="Show all presets side by side")
    args = parser.parse_args()

    img = cv2.imread(args.input)
    if img is None:
        print(f"ERROR: Could not open '{args.input}'")
        return

    if args.compare_all:
        # Compare all 3 presets
        results = []
        for preset_name in ["gentle", "default", "aggressive", "fast"]:
            enh = enhance(img, preset=preset_name)
            results.append((preset_name, enh))

        # Build comparison panel
        h    = img.shape[0]
        div  = np.full((h, 3, 3), 80, dtype=np.uint8)
        row  = img.copy()
        cv2.putText(row, "ORIGINAL", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)

        for name, enh in results:
            cv2.putText(enh, name.upper(), (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100,220,100), 1)
            row = np.hstack([row, div, enh])

        cv2.imshow("SphinxEyes — Compare all presets  (Q to quit)", row)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # Main pipeline
    enhanced = enhance(img, preset=args.preset)

    # Save
    if args.output:
        out_path = args.output
    else:
        p   = Path(args.input)
        out_path = str(p.with_stem(p.stem + '_enhanced').with_suffix('.jpg'))

    cv2.imwrite(out_path, enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  ✓ Saved: {out_path}  (preset: {args.preset})")

    if args.show:
        comp = compare_side_by_side(img, enhanced, label=args.preset)
        cv2.imshow("Original vs Enhanced  (Q to quit)", comp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()