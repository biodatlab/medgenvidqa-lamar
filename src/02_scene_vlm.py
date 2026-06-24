import os
import json
import glob
import re
import subprocess
import argparse
import torch
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

def format_timecode(time_str):
    """Keeps the timecode in standard HH:MM:SS format for JSON."""
    return time_str.split('.')[0]

def extract_video_segments(video_path, output_dir):
    """Finds scenes, uses -c copy, and tracks short/corrupted scenes."""
    print(f"\nDetecting scenes in {video_path}...")
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=27.0))
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()

    os.makedirs(output_dir, exist_ok=True)

    segment_data = []
    print(f"Found {len(scene_list)} scenes. Running Pass 1 (-c copy)...")

    for i, scene in enumerate(scene_list):
        start_time = scene[0].get_timecode()
        end_time = scene[1].get_timecode()
        duration_sec = scene[1].get_seconds() - scene[0].get_seconds()

        output_filename = os.path.join(output_dir, f"scene_{i+1:03d}.mp4")

        seg_info = {
            "index": i,
            "start_time": start_time,
            "end_time": end_time,
            "video_path": None,
            "duration": duration_sec
        }

        if duration_sec < 1.5:
            segment_data.append(seg_info)
            continue

        if not (os.path.exists(output_filename) and os.path.getsize(output_filename) > 1000):
            cmd = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-i', video_path, '-ss', start_time, '-to', end_time,
                '-c', 'copy', output_filename
            ]
            subprocess.run(cmd)

        if os.path.exists(output_filename) and os.path.getsize(output_filename) > 1000:
            seg_info["video_path"] = output_filename

        segment_data.append(seg_info)

    return segment_data

def describe_segments_with_qwen(video_path, segment_data, temp_dir, output_file, model, processor):
    """Runs Qwen, saves to JSONL, and merges adjacent skips for a second try."""
    print(f"\n--- Generating and Saving Chapters to {output_file} ---")

    if os.path.exists(output_file) and os.path.getsize(output_file) > 100:
        print(f"JSONL already exists for this video. Overwriting/restarting...")

    with open(output_file, "w", encoding="utf-8") as f:
        pass # Clear file

    failed_segments = []

    for seg in segment_data:
        if seg["video_path"] is None:
            failed_segments.append(seg)
            continue

        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": seg["video_path"], "max_pixels": 360 * 420, "fps": 1.0},
                        {"type": "text", "text": "You are a medical expert. Watch this short video segment and provide a description for what is happening. Keep it concise and descriptive."}
                    ],
                }
            ]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt"
            ).to("cuda")

            generated_ids = model.generate(**inputs, max_new_tokens=50)
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            record = {
                "raw_start_time": seg["start_time"],
                "start_timestamp": format_timecode(seg["start_time"]),
                "stop_timestamp": format_timecode(seg["end_time"]),
                "description": output_text.strip()
            }

            print(f"{record['start_timestamp']} - {record['stop_timestamp']} - {record['description']}")
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

        except Exception as e:
            print(f"⚠️ Failed on {format_timecode(seg['start_time'])} - Corrupted clip. Queueing for Pass 2.")
            failed_segments.append(seg)

    if failed_segments:
        print(f"\n--- PASS 2: Merging {len(failed_segments)} skipped/failed segments ---")

        groups = []
        current_group = []
        for seg in failed_segments:
            if not current_group:
                current_group.append(seg)
            elif seg["index"] == current_group[-1]["index"] + 1:
                current_group.append(seg)
            else:
                groups.append(current_group)
                current_group = [seg]
        if current_group:
            groups.append(current_group)

        for g_idx, group in enumerate(groups):
            merged_start = group[0]["start_time"]
            merged_end = group[-1]["end_time"]
            combined_filename = os.path.join(temp_dir, f"combined_retry_{g_idx:02d}.mp4")

            print(f"Retrying merged block: {format_timecode(merged_start)} to {format_timecode(merged_end)}")

            cmd = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-ss', merged_start, '-i', video_path, '-to', merged_end,
                '-c:v', 'libx264', '-preset', 'ultrafast', combined_filename
            ]
            subprocess.run(cmd)

            if os.path.exists(combined_filename):
                try:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "video", "video": combined_filename, "max_pixels": 360 * 420, "fps": 1.0},
                                {"type": "text", "text": "You are a medical expert. Watch this short video segment and provide a description for what is happening. Keep it concise and descriptive."}
                            ],
                        }
                    ]

                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")
                    generated_ids = model.generate(**inputs, max_new_tokens=50)
                    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

                    record = {
                        "raw_start_time": merged_start,
                        "start_timestamp": format_timecode(merged_start),
                        "stop_timestamp": format_timecode(merged_end),
                        "description": output_text.strip()
                    }

                    print(f"SUCCESS (Merged): {record['start_timestamp']} - {record['stop_timestamp']} - {record['description']}")
                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record) + "\n")
                except Exception as e:
                    print(f"❌ Permanent failure on merged segment {format_timecode(merged_start)}")

    print("\nSorting JSONL file chronologically...")
    with open(output_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    records = [json.loads(line) for line in lines]
    records.sort(key=lambda x: x["raw_start_time"])

    with open(output_file, "w", encoding="utf-8") as f:
        for r in records:
            if "raw_start_time" in r:
                del r["raw_start_time"]
            f.write(json.dumps(r) + "\n")

    print(f"Done with this video! Saved JSONL to '{output_file}'.")


def main():
    parser = argparse.ArgumentParser(description="Extract scenes and generate VLM descriptions.")
    parser.add_argument("--video_dir", type=str, required=True, help="Path to input raw videos directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save scene outputs and JSONL files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    search_pattern = os.path.join(args.video_dir, "*.mp4")
    video_files = glob.glob(search_pattern)

    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

    video_files.sort(key=natural_sort_key)
    print(f"Found {len(video_files)} videos to process in {args.video_dir}.")

    if not video_files:
        print("No videos found. Exiting.")
        return

    print(f"\nLoading model {MODEL_ID} into memory... (This will only happen once!)")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print("Model loaded successfully!")

    for video_path in video_files:
        video_name = os.path.basename(video_path)
        video_base = os.path.splitext(video_name)[0]

        print(f"\n" + "="*50)
        print(f"PROCESSING BATCH VIDEO: {video_name}")
        print("="*50)

        # Temp folder for the split mp4 chunks
        temp_scene_folder = os.path.join(args.output_dir, f"temp_scenes_{video_base}")
        # Final JSONL output path
        jsonl_output_path = os.path.join(args.output_dir, f"{video_base}_video_descriptions.jsonl")

        try:
            segments = extract_video_segments(video_path, temp_scene_folder)
            describe_segments_with_qwen(video_path, segments, temp_scene_folder, jsonl_output_path, model, processor)
        except Exception as e:
            print(f"\nCRITICAL ERROR processing {video_name}: {e}")
            print("Skipping to the next video...")
            continue

    print("\n" + "="*50)
    print("ALL VIDEOS PROCESSED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
