"""
Confidence scoring utilities for infant ear landmark predictions.
Provides methods to assess prediction confidence and filter landmarks accordingly.
"""

import numpy as np
import torch
import cv2
from scipy.stats import entropy


def calculate_heatmap_peak(heatmap):
    """
    Calculate the peak value of a heatmap.
    Higher peak indicates higher confidence in the landmark location.
    
    Args:
        heatmap: numpy array of shape (H, W) representing a single landmark heatmap
        
    Returns:
        float: Maximum value in the heatmap (0.0 to 1.0)
    """
    return float(np.max(heatmap))


def calculate_heatmap_entropy(heatmap):
    """
    Calculate the entropy of a heatmap's probability distribution.
    Lower entropy indicates higher confidence (concentrated probability).
    
    Args:
        heatmap: numpy array of shape (H, W)
        
    Returns:
        float: Entropy value. Lower is more confident.
    """
    # Normalize heatmap to probability distribution
    heatmap_norm = heatmap / (np.sum(heatmap) + 1e-8)
    heatmap_flat = heatmap_norm.flatten()
    
    # Calculate entropy (exclude near-zero values to reduce noise)
    heatmap_flat = heatmap_flat[heatmap_flat > 1e-4]
    
    if len(heatmap_flat) == 0:
        return float('inf')
    
    return float(entropy(heatmap_flat))


def calculate_distance_confidence(distance, max_distance=0.1, invert=True):
    """
    Calculate confidence score based on Euclidean distance error.
    
    Args:
        distance: float, Euclidean distance error in normalized coordinates (0-1)
        max_distance: float, distance considered as "maximum error" for normalization (default: 0.1)
        invert: bool, if True lower distance = higher confidence (default: True)
        
    Returns:
        float: Confidence score between 0.0 and 1.0
    """
    # Normalize distance to 0-1 range
    distance_norm = min(distance / max_distance, 1.0)
    
    if invert:
        # Lower distance = higher confidence
        confidence = 1.0 - distance_norm
    else:
        confidence = distance_norm
    
    return float(confidence)


def calculate_confidence_score(heatmap, peak_weight=0.7, entropy_weight=0.3, 
                               distance=None, distance_weight=0.0, max_distance=0.1):
    """
    Calculate combined confidence score for a landmark heatmap.
    
    Combines peak value, entropy, and optionally distance error to create a robust confidence metric.
    - Peak value: Higher is better (measures amplitude)
    - Entropy: Lower is better (measures concentration)
    - Distance: Lower is better (proximity to ground truth, requires ground truth labels)
    
    Args:
        heatmap: numpy array of shape (H, W)
        peak_weight: Weight for peak value component (default: 0.7)
        entropy_weight: Weight for entropy component (default: 0.3)
        distance: float, Euclidean distance error (optional, for validation/evaluation only)
        distance_weight: Weight for distance component (default: 0.0, no distance)
        max_distance: Distance considered as "maximum error" for normalization (default: 0.1)
        
    Returns:
        float: Confidence score between 0.0 and 1.0
    """
    peak = calculate_heatmap_peak(heatmap)
    h_entropy = calculate_heatmap_entropy(heatmap)
    
    # Normalize entropy component (lower entropy = higher confidence)
    # Cap entropy at reasonable max value
    max_entropy = 10.0
    entropy_norm = 1.0 - min(h_entropy / max_entropy, 1.0)
    
    # Start with heatmap-based confidence
    heatmap_confidence = (peak_weight * peak) + (entropy_weight * entropy_norm)
    
    # Optionally blend in distance-based confidence
    if distance is not None and distance_weight > 0.0:
        distance_conf = calculate_distance_confidence(distance, max_distance=max_distance)
        
        # Normalize weights to sum to 1
        total_heatmap_weight = peak_weight + entropy_weight
        confidence = (heatmap_confidence * (1.0 - distance_weight) + 
                     distance_conf * distance_weight)
    else:
        confidence = heatmap_confidence
    
    # Final normalization to 0-1 range
    confidence = np.clip(confidence, 0.0, 1.0)
    
    return float(confidence)


def get_confidence_for_landmarks(heatmaps, distances=None, distance_weight=0.0, max_distance=0.1):
    """
    Calculate confidence scores for all landmark heatmaps.
    
    Args:
        heatmaps: numpy array of shape (num_landmarks, H, W) or torch tensor
        distances: optional numpy array of shape (num_landmarks,) with Euclidean distance errors
        distance_weight: Weight for distance component (default: 0.0, no distance)
        max_distance: Distance considered as "maximum error" for normalization (default: 0.1)
        
    Returns:
        numpy array of shape (num_landmarks,) with confidence scores
    """
    if isinstance(heatmaps, torch.Tensor):
        heatmaps = heatmaps.cpu().numpy()
    
    num_landmarks = heatmaps.shape[0]
    confidences = np.zeros(num_landmarks)
    
    for i in range(num_landmarks):
        distance = distances[i] if distances is not None else None
        confidences[i] = calculate_confidence_score(
            heatmaps[i],
            distance=distance,
            distance_weight=distance_weight,
            max_distance=max_distance
        )
    
    return confidences


def get_total_confidence(confidences, metric='mean'):
    """
    Calculate aggregate confidence score across all landmarks.
    
    Args:
        confidences: numpy array of shape (num_landmarks,)
        metric: 'mean' (average), 'median' (middle value), or 'min' (worst landmark)
        
    Returns:
        float: Aggregate confidence score (0.0 to 1.0)
    """
    if metric == 'mean':
        return float(np.mean(confidences))
    elif metric == 'median':
        return float(np.median(confidences))
    elif metric == 'min':
        return float(np.min(confidences))
    else:
        raise ValueError(f"Unknown metric: {metric}")


def should_predict(confidences, threshold=0.5, metric='mean'):
    """
    Check if total landmark confidence is high enough to make a prediction.
    
    Args:
        confidences: numpy array of shape (num_landmarks,)
        threshold: Minimum confidence threshold (0.0 to 1.0)
        metric: 'mean' (average), 'median' (middle value), or 'min' (worst landmark)
        
    Returns:
        bool: True if total confidence >= threshold, False otherwise
    """
    total_conf = get_total_confidence(confidences, metric=metric)
    return bool(total_conf >= threshold)


def filter_landmarks_by_confidence(landmarks, confidences, threshold=0.5):
    """
    Filter landmarks based on confidence threshold.
    
    Args:
        landmarks: numpy array of shape (num_landmarks, 2) in normalized coordinates
        confidences: numpy array of shape (num_landmarks,)
        threshold: Confidence threshold (0.0 to 1.0)
        
    Returns:
        dict with keys:
            - 'high_conf': landmarks and indices above threshold
            - 'low_conf': landmarks and indices below threshold
    """
    high_conf_mask = confidences >= threshold
    low_conf_mask = ~high_conf_mask
    
    return {
        'high_conf': {
            'landmarks': landmarks[high_conf_mask],
            'indices': np.where(high_conf_mask)[0],
            'confidences': confidences[high_conf_mask]
        },
        'low_conf': {
            'landmarks': landmarks[low_conf_mask],
            'indices': np.where(low_conf_mask)[0],
            'confidences': confidences[low_conf_mask]
        }
    }


def get_confidence_color(confidence, confidence_style='distinct',
                         high_conf_threshold=0.8, medium_conf_threshold=0.6):

    """
    Get color for drawing landmarks based on confidence level.
    
    Args:
        confidence: float between 0.0 and 1.0
        confidence_style: 'distinct' for clear separation, 'gradient' for smooth gradient
        high_conf_threshold: threshold above which landmark is high confidence (default: 0.8)
        medium_conf_threshold: threshold above which landmark is medium confidence (default: 0.6)
        
    Returns:
        tuple: (B, G, R) color in OpenCV format
    """
    if confidence_style == 'distinct':
        if confidence >= high_conf_threshold:
            # High confidence: Bright green
            return (0, 255, 0)
        elif confidence >= medium_conf_threshold:
            # Medium confidence: Yellow
            return (0, 255, 255)
        else:
            # Low confidence: Red
            return (0, 0, 255)
    
    elif confidence_style == 'gradient':
        # Smooth color gradient from red -> yellow -> green
        if confidence < 0.5:
            # Red to Yellow gradient
            g = int(255 * (confidence * 2))  # 0 to 255
            r = 0
            b = 0
            return (b, g, r)
        else:
            # Yellow to Green gradient
            r = int(255 * (1 - (confidence - 0.5) * 2))  # 255 to 0
            g = 255
            b = 0
            return (b, g, r)
    
    return (128, 128, 128)  # Default gray


def draw_landmarks_with_confidence(image, landmarks_px, confidences, 
                                   high_conf_threshold=0.8,
                                   medium_conf_threshold=0.6,
                                   low_conf_threshold=0.4,
                                   confidence_style='distinct',
                                   show_confidence_text=True):
    """
    Draw landmarks on image with confidence-based visualization.

    All dots are the same size (radius=6, filled). Only color differs:
      Green  : >= 0.8
      Yellow : 0.6 – 0.8
      Red    : 0.4 – 0.6
      (skipped) : < 0.4

    Every visible dot gets a "idx:conf" label regardless of confidence level.

    Args:
        image: OpenCV image (H, W, 3)
        landmarks_px: numpy array of shape (num_landmarks, 2) in pixel coordinates
        confidences: numpy array of shape (num_landmarks,)
        high_conf_threshold: Threshold for high confidence (default: 0.8)
        medium_conf_threshold: Threshold for medium confidence (default: 0.6)
        low_conf_threshold: Threshold below which landmark is 
        confidence_style: 'distinct' or 'gradient'
        show_confidence_text: Whether to show "idx:conf" label next to every dot

    Returns:
        image: Modified image with drawn landmarks
    """
    img = image.copy()

    RADIUS    = 6   # same for all dots
    THICKNESS = -1  # filled

    for idx, (landmark, confidence) in enumerate(zip(landmarks_px, confidences)):
        x, y = int(landmark[0]), int(landmark[1])

        # Skip landmarks below the minimum threshold
        if confidence < low_conf_threshold:
            continue

        color = get_confidence_color(confidence, confidence_style,
                                     high_conf_threshold=high_conf_threshold,
                                     medium_conf_threshold=medium_conf_threshold)

        # Draw filled circle — same size for every landmark
        cv2.circle(img, (x, y), RADIUS, color, THICKNESS)

        # Label every visible dot with index and confidence score
        if show_confidence_text:
            text = f"{idx}:{confidence:.2f}"
            cv2.putText(img, text, (x + 9, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return img


def print_confidence_statistics(confidences, threshold_high=0.8, threshold_low=0.6):
    """
    Print detailed confidence statistics.
    
    Args:
        confidences: numpy array of confidence scores
        threshold_high: Threshold for high confidence
        threshold_low: Threshold for low confidence
    """
    high_conf = np.sum(confidences >= threshold_high)
    medium_conf = np.sum((confidences >= threshold_low) & (confidences < threshold_high))
    low_conf = np.sum(confidences < threshold_low)
    
    print(f"\n{'='*60}")
    print(f"Confidence Statistics:")
    print(f"{'='*60}")
    print(f"Total landmarks: {len(confidences)}")
    print(f"High confidence (>= {threshold_high}): {high_conf} ({100*high_conf/len(confidences):.1f}%)")
    print(f"Medium confidence ({threshold_low}-{threshold_high}): {medium_conf} ({100*medium_conf/len(confidences):.1f}%)")
    print(f"Low confidence (< {threshold_low}): {low_conf} ({100*low_conf/len(confidences):.1f}%)")
    print(f"\nMean confidence: {np.mean(confidences):.3f}")
    print(f"Std confidence:  {np.std(confidences):.3f}")
    print(f"Min confidence:  {np.min(confidences):.3f}")
    print(f"Max confidence:  {np.max(confidences):.3f}")
    print(f"{'='*60}\n")