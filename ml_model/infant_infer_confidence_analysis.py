"""
Comprehensive inference script for infant ear landmark detection with confidence analysis.

This script:
2. Classifies images as HIGH/MEDIUM/LOW confidence based on total confidence
4. Generates statistics comparing high vs medium vs low confidence predictions
5. Provides visual inspection of landmark accuracy by confidence level

Confidence bands (total confidence):
  HIGH   >= 0.8
  MEDIUM >= 0.6 and < 0.8
  LOW    <  0.6

Usage:
    python infant_infer_confidence_analysis.py --model /home/UFAD/mansapatel/ear-abnormalities/ml_model/infant_ear_model_v5.pth --images /home/UFAD/angelali/ears/images/images --output inference_results --metric mean
"""

import os
import argparse
import cv2
import numpy as np
import torch
from pathlib import Path

from adult_model import get_model, soft_argmax_2d
from infant_confidence_utils import (
    get_confidence_for_landmarks,
    draw_landmarks_with_confidence,
    print_confidence_statistics,
    get_total_confidence,
    should_predict
)


def draw_legend(image, label, total_confidence, confidence_metric,
                lm_high, lm_medium, lm_low,
                total_high, total_medium):
    """
    Draw a fixed-size, fixed-position legend onto image.
    Only the title text and title colour change between HIGH / MEDIUM / LOW;
    every other line (font size, position, spacing) is identical across all bands.

    Total confidence bands (image-level):
      HIGH   >= 0.8
      MEDIUM >= 0.6 and < 0.8
      LOW    <  0.6

    """
    # Title colour per band
    if total_confidence >= total_high:
        title_color = (0, 220, 0)        # green
        band_text   = f">= {total_high}"
    elif total_confidence >= total_medium:
        title_color = (0, 200, 255)      # yellow-orange
        band_text   = f"{total_medium} – {total_high}"
    else:
        title_color = (0, 60, 255)       # red
        band_text   = f"< {total_medium}"

    label_upper = label.upper()

    # # Semi-transparent dark background so text is readable on any image
    # overlay = image.copy()
    # cv2.rectangle(overlay, (5, 5), (430, 155), (0, 0, 0), -1)
    # cv2.addWeighted(overlay, 0.45, image, 0.55, 0, image)

    FONT       = cv2.FONT_HERSHEY_SIMPLEX
    TITLE_SCALE = 0.55
    BODY_SCALE  = 0.42
    TITLE_THICK = 2
    BODY_THICK  = 1
    GRAY        = (180, 180, 180)

    # Line 1 – title
    cv2.putText(image,
                f"{label_upper} CONFIDENCE  ({band_text})  |  score: {total_confidence:.3f}",
                (10, 38), FONT, TITLE_SCALE, title_color, TITLE_THICK)

    # # Line 2-5 – landmark dot legend (identical across all three bands)
    # cv2.putText(image,
    #             f"Green  : >= {lm_high}",
    #             (10, 62), FONT, BODY_SCALE, (0, 255, 0), BODY_THICK)
    # cv2.putText(image,
    #             f"Yellow : {lm_medium} - {lm_high}",
    #             (10, 82), FONT, BODY_SCALE, (0, 255, 255), BODY_THICK)
    # cv2.putText(image,
    #             f"Red    : {lm_low} - {lm_medium}",
    #             (10, 102), FONT, BODY_SCALE, (0, 60, 255), BODY_THICK)
    # cv2.putText(image,
    #             f"(skipped) : < {lm_low}",
    #             (10, 122), FONT, BODY_SCALE, GRAY, BODY_THICK)

    # # Line 6 – metric footer
    # cv2.putText(image,
    #             f"Metric: {confidence_metric}",
    #             (10, 142), FONT, BODY_SCALE, GRAY, BODY_THICK)

    return image


def save_confidence_report(output_dir, high_conf_imgs, medium_conf_imgs, low_conf_imgs, statistics):

    """Save a detailed confidence report."""
    report_path = os.path.join(output_dir, "confidence_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("INFANT EAR LANDMARK CONFIDENCE ANALYSIS REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"HIGH CONFIDENCE PREDICTIONS (>= {statistics['high_total_threshold']}): {len(high_conf_imgs)}\n")
        f.write("-" * 70 + "\n")
        for img_name, conf in high_conf_imgs:
            f.write(f"  {img_name:40s} | Total Conf: {conf:.3f}\n")
        
        f.write(f"\n\nMEDIUM CONFIDENCE PREDICTIONS (>= {statistics['medium_total_threshold']} and < {statistics['high_total_threshold']}): {len(medium_conf_imgs)}\n")
        f.write("-" * 70 + "\n")
        for img_name, conf in medium_conf_imgs:
            f.write(f"  {img_name:40s} | Total Conf: {conf:.3f}\n")
        
        f.write(f"\n\nLOW CONFIDENCE PREDICTIONS (< {statistics['medium_total_threshold']}): {len(low_conf_imgs)}\n")
        f.write("-" * 70 + "\n")
        for img_name, conf in low_conf_imgs:
            f.write(f"  {img_name:40s} | Total Conf: {conf:.3f}\n")
        
        f.write(f"\n\n{'='*70}\n")
        f.write("STATISTICS\n")
        f.write(f"{'='*70}\n")
        f.write(f"Total images processed: {statistics['total']}\n")
        
        if statistics['total'] > 0:
            f.write(f"High confidence (>= {statistics['high_total_threshold']}): {statistics['high_count']} ({100*statistics['high_count']/statistics['total']:.1f}%)\n")
            f.write(f"Medium confidence ({statistics['medium_total_threshold']} to {statistics['high_total_threshold']}): {statistics['medium_count']} ({100*statistics['medium_count']/statistics['total']:.1f}%)\n")
            f.write(f"Low confidence (< {statistics['medium_total_threshold']}): {statistics['low_count']} ({100*statistics['low_count']/statistics['total']:.1f}%)\n")
        else:
            f.write(f"[ERROR] No images were processed. Check the images directory path.\n")
        
        f.write(f"\nThresholds used:\n")
        f.write(f"  Total confidence HIGH   : >= {statistics['high_total_threshold']}\n")
        f.write(f"  Total confidence MEDIUM : >= {statistics['medium_total_threshold']} and < {statistics['high_total_threshold']}\n")
        f.write(f"  Total confidence LOW    : < {statistics['medium_total_threshold']}\n")
        f.write(f"  Individual landmark HIGH: {statistics['high_lm_threshold']}\n")
        f.write(f"  Individual landmark MED : {statistics['medium_lm_threshold']}\n")
        f.write(f"  Individual landmark LOW : {statistics['low_lm_threshold']}\n")
        f.write(f"  Total confidence metric : {statistics['confidence_metric']}\n")
    
    print(f"✓ Report saved to {report_path}")


def infer_on_directory(model_path, images_dir, output_dir, confidence_metric='mean',
                       num_landmarks=23, num_stages=6):
    """
    Run inference on all images in a directory with confidence analysis.
    
    Total confidence bands:
        HIGH   >= 0.8
        MEDIUM >= 0.6 and < 0.8
        LOW    <  0.6

    Args:
        model_path: Path to trained model checkpoint
        images_dir: Directory containing images
        output_dir: Output directory for results
        confidence_metric: 'mean', 'median', or 'min' for total confidence
        num_landmarks: Number of landmarks (default: 23)
        num_stages: Number of stages in model (default: 6)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    model = get_model(num_landmarks, num_stages).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✓ Loaded model: {model_path}")
    
    # Create output subdirectories
    high_conf_dir = os.path.join(output_dir, "high_confidence")
    medium_conf_dir = os.path.join(output_dir, "medium_confidence")
    low_conf_dir = os.path.join(output_dir, "low_confidence")
    stats_dir = os.path.join(output_dir, "statistics")
    
    os.makedirs(high_conf_dir, exist_ok=True)
    os.makedirs(medium_conf_dir, exist_ok=True)
    os.makedirs(low_conf_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)
    
    print(f"✓ Output directories created")
    print(f"  - High confidence  : {high_conf_dir}")
    print(f"  - Medium confidence: {medium_conf_dir}")
    print(f"  - Low confidence   : {low_conf_dir}")
    print(f"  - Statistics       : {stats_dir}")


    TOTAL_HIGH_THRESHOLD   = 0.8   # >= 0.8  → HIGH
    TOTAL_MEDIUM_THRESHOLD = 0.6   # >= 0.6  → MEDIUM  (< 0.8)
    #                                # < 0.6   → LOW

    LM_HIGH_THRESHOLD   = 0.8
    LM_MEDIUM_THRESHOLD = 0.6
    LM_LOW_THRESHOLD    = 0.4
    
    # Collect results
    high_conf_imgs = []
    medium_conf_imgs = []
    low_conf_imgs = []
    all_confidences = []
    all_total_confidences = []
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [f for f in os.listdir(images_dir) 
                   if os.path.splitext(f.lower())[1] in image_extensions]
    
    if not image_files:
        print(f"\n✗ ERROR: No images found in {images_dir}")
        print(f"✗ Checked for extensions: {image_extensions}")
        print(f"✗ Directory contents: {os.listdir(images_dir)[:10]}")
        return
    
    print(f"✓ Found {len(image_files)} images to process\n")
    
    with torch.no_grad():
        for idx, img_name in enumerate(image_files):
            img_path = os.path.join(images_dir, img_name)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"[{idx+1}/{len(image_files)}] ✗ Failed to load: {img_name}")
                continue
            
            h, w = img.shape[:2]
            
            # Prepare image for model
            img_resized = cv2.resize(img, (368, 368))
            img_normalized = img_resized / 255.0
            img_tensor = np.transpose(img_normalized, (2, 0, 1))
            img_tensor = torch.tensor(img_tensor, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Get predictions
            pred_heatmaps = model(img_tensor)[:, -1]
            pred = soft_argmax_2d(pred_heatmaps, normalize=True).cpu().numpy().squeeze()
            
            # Calculate confidence
            confidences = get_confidence_for_landmarks(pred_heatmaps.squeeze())
            all_confidences.extend(confidences)
            
            # Get total confidence
            total_confidence = get_total_confidence(confidences, metric=confidence_metric)
            all_total_confidences.append(total_confidence)
            
            # DEBUG: Print confidence distribution for first few images
            if idx < 2:
                heatmaps_np = pred_heatmaps.squeeze().cpu().numpy()
                print(f"\nDEBUG {img_name}:")
                print(f"  Heatmap ranges - min: {np.min(heatmaps_np):.6f}, max: {np.max(heatmaps_np):.6f}")
                print(f"  Landmark confidences - min={np.min(confidences):.3f}, max={np.max(confidences):.3f}, mean={np.mean(confidences):.3f}")
                print(f"  LM HIGH (>={LM_HIGH_THRESHOLD}): {np.sum(confidences >= LM_HIGH_THRESHOLD)}/{len(confidences)}")
                print(f"  LM MED  ({LM_MEDIUM_THRESHOLD}-{LM_HIGH_THRESHOLD}): {np.sum((confidences >= LM_MEDIUM_THRESHOLD) & (confidences < LM_HIGH_THRESHOLD))}/{len(confidences)}")
                print(f"  LM LOW  (<{LM_MEDIUM_THRESHOLD}): {np.sum(confidences < LM_MEDIUM_THRESHOLD)}/{len(confidences)}")
                print(f"  Total confidence ({confidence_metric}): {total_confidence:.3f}")
            
            # Prepare output image
            img_output = img.copy()
            pred_px = pred * np.array([w - 1, h - 1])

            img_output = draw_landmarks_with_confidence(
                img_output, pred_px, confidences,
                high_conf_threshold=LM_HIGH_THRESHOLD,
                medium_conf_threshold=LM_MEDIUM_THRESHOLD,
                low_conf_threshold=LM_LOW_THRESHOLD,
                confidence_style='distinct',
                show_confidence_text=True
            )

            # Route and draw legend — only the title colour/text changes
            if total_confidence >= TOTAL_HIGH_THRESHOLD:
                out_path = os.path.join(high_conf_dir, img_name)
                high_conf_imgs.append((img_name, total_confidence))
                status = "✓ HIGH"
                label = "high"
            elif total_confidence >= TOTAL_MEDIUM_THRESHOLD:
                out_path = os.path.join(medium_conf_dir, img_name)
                medium_conf_imgs.append((img_name, total_confidence))
                status = "~ MEDIUM"
                label = "medium"
            else:
                out_path = os.path.join(low_conf_dir, img_name)
                low_conf_imgs.append((img_name, total_confidence))
                status = "✗ LOW"
                label = "low"

            img_output = draw_legend(
                img_output, label, total_confidence, confidence_metric,
                lm_high=LM_HIGH_THRESHOLD,
                lm_medium=LM_MEDIUM_THRESHOLD,
                lm_low=LM_LOW_THRESHOLD,
                total_high=TOTAL_HIGH_THRESHOLD,
                total_medium=TOTAL_MEDIUM_THRESHOLD,
            )
            
            cv2.imwrite(out_path, img_output)
            print(f"[{idx+1}/{len(image_files)}] {status} | {img_name:40s} | Conf: {total_confidence:.3f}")
    

    statistics = {
        'total': len(image_files),
        'high_count': len(high_conf_imgs),
        'medium_count': len(medium_conf_imgs),
        'low_count': len(low_conf_imgs),
        'high_total_threshold': TOTAL_HIGH_THRESHOLD,
        'medium_total_threshold': TOTAL_MEDIUM_THRESHOLD,
        'confidence_metric': confidence_metric,
        'high_lm_threshold': LM_HIGH_THRESHOLD,
        'medium_lm_threshold': LM_MEDIUM_THRESHOLD,
        'low_lm_threshold': LM_LOW_THRESHOLD,
    }
    

    save_confidence_report(output_dir, high_conf_imgs, medium_conf_imgs, low_conf_imgs, statistics)
    
    # Print summary statistics
    all_confidences = np.array(all_confidences)
    all_total_confidences = np.array(all_total_confidences)
    
    if len(image_files) == 0:
        print("\n✗ No images were processed. Exiting.")
        return
    
    print("\n" + "="*70)
    print("INFERENCE COMPLETE - CONFIDENCE ANALYSIS SUMMARY")
    print("="*70)
    print(f"\nTotal images processed: {len(image_files)}")
    print(f"High   confidence (>= {TOTAL_HIGH_THRESHOLD}):                    {len(high_conf_imgs)} ({100*len(high_conf_imgs)/len(image_files):.1f}%)")
    print(f"Medium confidence (>= {TOTAL_MEDIUM_THRESHOLD} and < {TOTAL_HIGH_THRESHOLD}): {len(medium_conf_imgs)} ({100*len(medium_conf_imgs)/len(image_files):.1f}%)")
    print(f"Low    confidence (<  {TOTAL_MEDIUM_THRESHOLD}):                    {len(low_conf_imgs)} ({100*len(low_conf_imgs)/len(image_files):.1f}%)")
    print(f"\nConfidence metric used: {confidence_metric}")
    print(f"\nIndividual landmark confidence:")
    print_confidence_statistics(all_confidences, threshold_high=LM_HIGH_THRESHOLD,
                                threshold_low=LM_LOW_THRESHOLD)
    
    print(f"\nTotal prediction confidence ({confidence_metric}) across all images:")
    print(f"  Mean:   {np.mean(all_total_confidences):.3f}")
    print(f"  Median: {np.median(all_total_confidences):.3f}")
    print(f"  Std:    {np.std(all_total_confidences):.3f}")
    print(f"  Min:    {np.min(all_total_confidences):.3f}")
    print(f"  Max:    {np.max(all_total_confidences):.3f}")
    
    print(f"\n✓ Results saved to {output_dir}")
    print(f"  → High   confidence: {len(high_conf_imgs)} images → {high_conf_dir}")
    print(f"  → Medium confidence: {len(medium_conf_imgs)} images → {medium_conf_dir}")
    print(f"  → Low    confidence: {len(low_conf_imgs)} images → {low_conf_dir}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Infant ear landmark inference with confidence analysis'
    )
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--images', type=str, required=True,
                       help='Directory containing images')
    parser.add_argument('--output', type=str, default='inference_results',
                       help='Output directory for results')
    parser.add_argument('--metric', type=str, default='mean',
                       choices=['mean', 'median', 'min'],
                       help='Metric for total confidence (mean/median/min)')
    parser.add_argument('--landmarks', type=int, default=23,
                       help='Number of landmarks')
    parser.add_argument('--stages', type=int, default=6,
                       help='Number of stages')
    
    args = parser.parse_args()
    
    infer_on_directory(
        args.model, args.images, args.output,
        confidence_metric=args.metric,
        num_landmarks=args.landmarks,
        num_stages=args.stages
    )