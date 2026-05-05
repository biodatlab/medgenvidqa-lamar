import os
import json
import torch
import subprocess
import argparse
from concurrent.futures import ThreadPoolExecutor
from qwen_asr import Qwen3ASRModel

def get_processed_ids(output_file):
    processed = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    processed.add(data['video_id'])
                except: continue
    return processed

def get_target_video_ids(test_json_file):
    target_ids = set()
    if os.path.exists(test_json_file):
        with open(test_json_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
            for item in test_data:
                vid = item.get("video_id", item.get("id"))
                if vid:
                    target_ids.add(vid)
    return target_ids

def format_word_timestamps(word_timestamps):
    words_list = []
    if not word_timestamps:
        return words_list

    for ts in word_timestamps:
        word = ts.text.strip()
        if not word:
            continue
        start_t = getattr(ts, 'start', getattr(ts, 'start_time', 0))
        end_t = getattr(ts, 'end', getattr(ts, 'end_time', 0))

        words_list.append({
            "word": word,
            "start": start_t,
            "end": end_t
        })
    return words_list

def convert_single_video(args):
    video_path, temp_wav_path = args
    try:
        if os.path.exists(temp_wav_path): os.remove(temp_wav_path)

        result = subprocess.run([
            "ffmpeg", "-y", "-i", video_path,
            "-ar", "16000", "-ac", "1", "-vn",
            temp_wav_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

        if result.returncode == 0 and os.path.exists(temp_wav_path) and os.path.getsize(temp_wav_path) > 1000:
            return temp_wav_path
        else:
            return None
    except:
        return None

def process_folder(model, video_dir, output_file, test_json_file, batch_size):
    processed_ids = get_processed_ids(output_file)
    target_ids = get_target_video_ids(test_json_file)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    temp_audio_dir = os.path.join(os.path.dirname(output_file), "temp_audio")
    os.makedirs(temp_audio_dir, exist_ok=True)

    all_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
    all_files.sort()

    files_to_process = []
    for filename in all_files:
        video_id = filename.replace(".mp4", "")
        if video_id in target_ids and video_id not in processed_ids:
            files_to_process.append(filename)

    with open(output_file, 'a', encoding='utf-8') as f_out:
        for i in range(0, len(files_to_process), batch_size):
            batch_files = files_to_process[i : i + batch_size]
            
            batch_ids = []
            conversion_tasks = []

            for filename in batch_files:
                video_id = filename.replace(".mp4", "")
                video_path = os.path.join(video_dir, filename)
                temp_wav = os.path.join(temp_audio_dir, f"{video_id}.wav")

                batch_ids.append(video_id)
                conversion_tasks.append((video_path, temp_wav))

            valid_audio_paths = []
            valid_indices = []

            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                converted_paths = list(executor.map(convert_single_video, conversion_tasks))

            for idx, path in enumerate(converted_paths):
                if path:
                    valid_audio_paths.append(path)
                    valid_indices.append(idx)

            if not valid_audio_paths:
                continue

            try:
                results = model.transcribe(
                    audio=valid_audio_paths,
                    language=["English"] * len(valid_audio_paths),
                    return_time_stamps=True,
                )

                for k, result_idx in enumerate(valid_indices):
                    video_id = batch_ids[result_idx]
                    full_text = results[k].text if results[k].text else ""
                    word_timestamps = results[k].time_stamps if hasattr(results[k], 'time_stamps') else []
                    formatted_words = format_word_timestamps(word_timestamps)

                    entry = {
                        "video_id": video_id,
                        "full_text": full_text,
                        "words": formatted_words
                    }

                    f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")

                    if os.path.exists(valid_audio_paths[k]):
                        os.remove(valid_audio_paths[k])

                f_out.flush()

            except Exception:
                pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Qwen3-ASR to generate word-level transcripts.")
    parser.add_argument("--video_dir", type=str, default="data/raw_videos/", help="Directory containing MP4 videos")
    parser.add_argument("--output_file", type=str, default="data/transcripts/qwen_transcriptions.jsonl", help="Path to save JSONL output")
    parser.add_argument("--test_json", type=str, default="data/queries/task_c_test.json", help="Path to the JSON file with target video IDs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for parallel processing")
    args = parser.parse_args()
    
    model = Qwen3ASRModel.LLM(
        model="Qwen/Qwen3-ASR-1.7B",
        gpu_memory_utilization=0.6,
        max_inference_batch_size=128,
        max_new_tokens=4096,
        forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
        enforce_eager=True,
        forced_aligner_kwargs=dict(
            dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation="flash_attention_2"
        ),
    )

    process_folder(model, args.video_dir, args.output_file, args.test_json, args.batch_size)