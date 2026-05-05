import os
import json
import requests
import argparse

def download_videos(json_file_path, output_folder):
    """
    Downloads mp4 videos from URLs provided in a JSON file.
    """
    if not os.path.exists(json_file_path):
        return

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            video_data = json.load(f)
    except json.JSONDecodeError:
        return

    os.makedirs(output_folder, exist_ok=True)

    success_count = 0
    fail_count = 0

    for item in video_data:
        video_id = item.get('id')
        video_url = item.get('video')

        if not video_id or not video_url:
            fail_count += 1
            continue

        filename = f"{video_id}.mp4"
        filepath = os.path.join(output_folder, filename)

        if os.path.exists(filepath):
            success_count += 1
            continue

        try:
            response = requests.get(video_url, stream=True, timeout=20)
            response.raise_for_status()

            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            success_count += 1

        except requests.exceptions.RequestException:
            fail_count += 1

    return success_count, fail_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download videos from MedGenVidQA JSON.")
    
    parser.add_argument("--json_path", type=str, default="data/queries/task_c_test.json", 
                        help="Path to the input JSON file containing video URLs.")
    parser.add_argument("--output_dir", type=str, default="data/raw_videos/", 
                        help="Directory to save the downloaded mp4 files.")
    
    args = parser.parse_args()

    download_videos(args.json_path, args.output_dir)