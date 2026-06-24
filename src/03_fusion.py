import os
import json
import argparse
import glob

def timecode_to_seconds(time_str):
    """Converts HH:MM:SS or MM:SS strings from the visual model into raw seconds."""
    parts = time_str.split(':')
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return 0.0

def load_all_transcripts(asr_dir):
    """Loads transcript files from the ASR directory into memory."""
    print(f"Loading transcript files from {asr_dir}...")
    all_transcripts = {}

    # Support both a single master file or a directory of individual JSON/JSONL files
    if os.path.isfile(asr_dir):
        files_to_process = [asr_dir]
    else:
        files_to_process = glob.glob(os.path.join(asr_dir, "*.json*"))

    if not files_to_process:
        print(f"CRITICAL ERROR: No transcript files found in {asr_dir}")
        return all_transcripts

    for file_path in files_to_process:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        video_id = data.get("video_id")
                        if video_id:
                            # Save the words list mapped to the video ID
                            all_transcripts[video_id] = data.get("words", [])
                    except json.JSONDecodeError:
                        continue

    print(f"Successfully loaded transcripts for {len(all_transcripts)} videos.")
    return all_transcripts

def merge_visual_and_audio(descriptions_path, transcript_words, output_path, video_id):
    """Merges the visual JSONL with the in-memory transcript list."""
    segments = []
    with open(descriptions_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                segments.append(json.loads(line))

    with open(output_path, 'w', encoding='utf-8') as f_out:
        for i, seg in enumerate(segments):

            # 1. Convert visual boundaries to raw seconds
            seg_start_sec = timecode_to_seconds(seg["start_timestamp"])
            seg_end_sec = timecode_to_seconds(seg["stop_timestamp"])

            # 2. Extract words and stitch them into a single string
            segment_words = []
            for word_data in transcript_words:
                word_start = float(word_data["start"])

                # If the word happens during this visual scene, grab the text
                if seg_start_sec <= word_start < seg_end_sec:
                    segment_words.append(word_data["word"])

            segment_transcript = " ".join(segment_words)

            # 3. Construct the final dataset object
            master_record = {
                "id": f"{video_id}_{i+1:03d}",
                "segment_start": seg["start_timestamp"],
                "segment_stop": seg["stop_timestamp"],
                "context": seg["description"],
                "transcript": segment_transcript
            }

            # Write to the final JSONL file
            f_out.write(json.dumps(master_record) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Multimodal Fusion: Aligns ASR transcripts with visual scene descriptions.")
    parser.add_argument("--asr_dir", type=str, required=True, help="Path to ASR JSONL transcript file or directory")
    parser.add_argument("--scene_dir", type=str, required=True, help="Path to directory containing extracted scenes and JSONL files")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the fused master context datasets")
    args = parser.parse_args()

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Pre-load all transcripts into memory
    master_transcript_dict = load_all_transcripts(args.asr_dir)

    # 2. Loop through all the YouTube video IDs found in the transcript file
    video_ids = list(master_transcript_dict.keys())

    if not video_ids:
        print("No transcripts loaded. Exiting.")
        return

    for video_id in video_ids:
        print(f"\nProcessing {video_id}...")

        # Construct dynamic paths based on the structure created by 02_scene_vlm.py
        descriptions_jsonl = os.path.join(args.scene_dir, f"extracted_scenes_{video_id}", f"{video_id}_video_descriptions.jsonl")
        output_master_jsonl = os.path.join(args.output_dir, f"{video_id}_master_dataset.jsonl")

        # Safety Check 1: Did Qwen successfully generate descriptions for this video?
        if not os.path.exists(descriptions_jsonl):
            print(f"⚠️ Skipped {video_id}: Descriptions file not found ({descriptions_jsonl})")
            continue

        # Safety Check 2: Does this video have a transcript in the master JSONL?
        transcript_words = master_transcript_dict.get(video_id)
        if not transcript_words:
            print(f"⚠️ Skipped {video_id}: No transcript words found in master transcript file.")
            continue

        # Run the merger
        try:
            merge_visual_and_audio(descriptions_jsonl, transcript_words, output_master_jsonl, video_id)
            print(f"✅ Success! Saved {video_id}_master_dataset.jsonl")
        except Exception as e:
            print(f"❌ Error merging {video_id}: {e}")

    print("\n" + "="*50)
    print(f"ALL {len(video_ids)} VIDEOS PROCESSED!")

if __name__ == "__main__":
    main()
