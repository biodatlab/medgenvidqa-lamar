import json
import os
import argparse
import glob

def time_to_seconds(time_str: str) -> int:
    """Converts a time string (MM:SS or HH:MM:SS) to total seconds."""
    if not time_str:
        return 0
    
    parts = str(time_str).strip().split(':')
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return 0

def calculate_iou(pred_start: int, pred_end: int, gt_start: int, gt_end: int) -> float:
    """Calculates Intersection over Union (IoU) for two 1D time segments."""
    intersect_start = max(pred_start, gt_start)
    intersect_end = min(pred_end, gt_end)
    intersection = max(0, intersect_end - intersect_start)

    pred_duration = max(0, pred_end - pred_start)
    gt_duration = max(0, gt_end - gt_start)
    union = pred_duration + gt_duration - intersection

    if union <= 0:
        return 0.0
    return intersection / union

def load_ground_truth(gt_path: str) -> dict:
    """Loads ground truth data, supporting both JSON arrays and JSONL."""
    ground_truth_data = {}
    if not os.path.exists(gt_path):
        print(f"⚠️ Error: Ground truth file not found at {gt_path}")
        return ground_truth_data

    with open(gt_path, 'r', encoding='utf-8') as f:
        try:
            # Attempt to parse as a standard JSON list/object
            data = json.load(f)
            if isinstance(data, dict) and "data" in data:
                data = data["data"]
            if isinstance(data, list):
                for record in data:
                    gt_id = record.get("id")
                    if gt_id:
                        ground_truth_data[gt_id] = {
                            "start": time_to_seconds(record.get("annotate start")),
                            "end": time_to_seconds(record.get("annotate end"))
                        }
                return ground_truth_data
        except json.JSONDecodeError:
            pass # Fallback to JSONL

    # Fallback: Parse as JSONL
    with open(gt_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): 
                continue
            try:
                record = json.loads(line)
                gt_id = record.get("id")
                if gt_id:
                    ground_truth_data[gt_id] = {
                        "start": time_to_seconds(record.get("annotate start")),
                        "end": time_to_seconds(record.get("annotate end"))
                    }
            except json.JSONDecodeError:
                continue
                
    return ground_truth_data

def load_predictions(pred_dir: str) -> list:
    """Loads prediction data from a directory of JSONs or a single JSON file."""
    predictions_data = []
    
    if os.path.isfile(pred_dir):
        files = [pred_dir]
    elif os.path.isdir(pred_dir):
        files = glob.glob(os.path.join(pred_dir, "*.json"))
    else:
        print(f"⚠️ Error: Prediction path not found at {pred_dir}")
        return predictions_data

    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    predictions_data.extend(data)
                elif isinstance(data, dict):
                    # Handle case where output is a single object or wrapped in a dict
                    if "id" in data:
                        predictions_data.append(data)
                    elif "data" in data:
                        predictions_data.extend(data["data"])
            except json.JSONDecodeError:
                print(f"⚠️ Warning: Could not parse {file_path}")
                
    return predictions_data

def evaluate_predictions(gt_path: str, pred_dir: str):
    """Evaluates prediction accuracy against ground truth data using IoU."""
    
    ground_truth_data = load_ground_truth(gt_path)
    if not ground_truth_data:
        return
        
    predictions_data = load_predictions(pred_dir)
    if not predictions_data:
        return

    # Metrics Tracking
    total_iou = 0.0
    valid_comparisons = 0
    iou_03_count = 0
    iou_05_count = 0
    iou_07_count = 0

    # Calculate IoU for matching IDs
    for pred in predictions_data:
        vid_id = pred.get("id")

        if vid_id in ground_truth_data:
            pred_start = time_to_seconds(pred.get("answer_start"))
            pred_end = time_to_seconds(pred.get("answer_end"))

            gt_start = ground_truth_data[vid_id]["start"]
            gt_end = ground_truth_data[vid_id]["end"]

            iou = calculate_iou(pred_start, pred_end, gt_start, gt_end)

            total_iou += iou
            valid_comparisons += 1

            # Update Threshold Counts
            if iou >= 0.3: iou_03_count += 1
            if iou >= 0.5: iou_05_count += 1
            if iou >= 0.7: iou_07_count += 1

    # Output Results
    if valid_comparisons > 0:
        miou = total_iou / valid_comparisons
        iou_03_pct = (iou_03_count / valid_comparisons) * 100
        iou_05_pct = (iou_05_count / valid_comparisons) * 100
        iou_07_pct = (iou_07_count / valid_comparisons) * 100

        print("=" * 40)
        print("🎯 EVALUATION RESULTS")
        print("=" * 40)
        print(f"Total Videos Evaluated : {valid_comparisons}")
        print(f"Mean IoU (mIoU)        : {miou:.4f}")
        print("-" * 40)
        print(f"IoU >= 0.3 Threshold   : {iou_03_pct:.2f}%  ({iou_03_count}/{valid_comparisons})")
        print(f"IoU >= 0.5 Threshold   : {iou_05_pct:.2f}%  ({iou_05_count}/{valid_comparisons})")
        print(f"IoU >= 0.7 Threshold   : {iou_07_pct:.2f}%  ({iou_07_count}/{valid_comparisons})")
    else:
        print("⚠️ No matching IDs found between Ground Truth and Predictions.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LAMAR-2 timestamp predictions against ground truth.")
    parser.add_argument("--pred_dir", type=str, required=True, help="Directory containing prediction JSON files (or path to a single JSON file).")
    parser.add_argument("--gt_path", type=str, required=True, help="Path to the ground truth JSON or JSONL file.")
    
    args = parser.parse_args()
    evaluate_predictions(args.gt_path, args.pred_dir)
