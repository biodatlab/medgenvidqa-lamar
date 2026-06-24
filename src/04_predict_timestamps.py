import os
import json
import time
import argparse
from google import genai

def get_processed_tasks(output_filepath):
    """Reads the output file and returns a set of (video_id, question) tuples already processed."""
    processed = set()
    if os.path.exists(output_filepath):
        with open(output_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        video_id = data.get("id")
                        question = data.get("question")
                        if video_id and question:
                            processed.add((video_id, question))
                    except json.JSONDecodeError:
                        continue
    return processed

def load_additional_data(jsonl_path):
    """Loads the master dataset for a specific video and formats it for the prompt."""
    if not os.path.exists(jsonl_path):
        return "[No additional context or transcription available for this video.]"

    formatted_text = ""
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                seg_start = data.get("segment_start", "Unknown")
                seg_end = data.get("segment_stop", "Unknown")
                context = data.get("context", "None")
                transcript = data.get("transcript", "None")

                formatted_text += f"Timestamp: [{seg_start} -> {seg_end}]\n"
                formatted_text += f"Visual Action: {context}\n"
                formatted_text += f"Spoken Transcript: {transcript}\n\n"
            except Exception:
                continue

    return formatted_text.strip()

def main():
    parser = argparse.ArgumentParser(description="Predict video timestamps using Gemini-3-Flash.")
    # Set default paths to match the README usage
    parser.add_argument("--query_file", type=str, default="data/queries/task_c_test.json", help="Path to JSON file containing queries")
    parser.add_argument("--video_dir", type=str, default="data/raw_videos/", help="Path to raw mp4 videos")
    parser.add_argument("--prompts_file", type=str, default="prompts/prompts.json", help="Path to the prompts JSON file")
    
    # Required arguments matching the bash command
    parser.add_argument("--context_dir", type=str, required=True, help="Path to fused master context JSONL files")
    parser.add_argument("--prompt_type", type=str, choices=["zero_shot", "strict", "cot", "heuristic_loose"], required=True, help="Which prompt template to use")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the prediction JSONL files")
    args = parser.parse_args()

    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("CRITICAL ERROR: GEMINI_API_KEY environment variable not set.")
    
    client = genai.Client(api_key=api_key)

    if not os.path.exists(args.query_file):
        print(f"❌ Query file not found: {args.query_file}")
        return

    if not os.path.exists(args.prompts_file):
        print(f"❌ Prompts file not found: {args.prompts_file}. Please ensure it exists.")
        return

    # Load prompt templates dynamically
    with open(args.prompts_file, 'r', encoding='utf-8') as f:
        PROMPTS = json.load(f)

    with open(args.query_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"predictions_{args.prompt_type}.jsonl")

    processed_tasks = get_processed_tasks(output_file)
    if processed_tasks:
        print(f"🔄 Resuming session: Found {len(processed_tasks)} already processed tasks.")
    else:
        print("▶️ Starting fresh session: No previous output found.")

    uploaded_videos_cache = {}

    with open(output_file, 'a', encoding='utf-8') as f_out:
        for item in test_data:
            video_id = item.get("id")
            question = item.get("question")

            if not video_id or not question: continue

            if (video_id, question) in processed_tasks:
                print(f"⏭️ Skipping already processed: {video_id} - '{question[:30]}...'")
                continue

            print(f"\nProcessing Video: {video_id} using {args.prompt_type} prompt")

            video_path = os.path.join(args.video_dir, f"{video_id}.mp4")
            
            if not os.path.exists(video_path):
                print(f"⚠️ Video file not found: {video_path}. Skipping.")
                continue

            # --- UPLOAD / CACHE VIDEO ---
            if video_id in uploaded_videos_cache:
                uploaded_video = uploaded_videos_cache[video_id]
            else:
                try:
                    print(f"⬆️ Uploading {video_id}.mp4 to Gemini servers", end="")
                    uploaded_video = client.files.upload(file=video_path)

                    while uploaded_video.state.name == "PROCESSING":
                        print(".", end="", flush=True)
                        time.sleep(5)
                        uploaded_video = client.files.get(name=uploaded_video.name)
                    print()

                    if uploaded_video.state.name == "FAILED":
                        print(f"❌ Video {video_id} failed to process on Google's servers. Skipping.")
                        continue

                    uploaded_videos_cache[video_id] = uploaded_video
                except Exception as e:
                    print(f"❌ Upload Failed: {e}")
                    continue

            # --- BUILD PROMPT ---
            if args.prompt_type == "zero_shot":
                final_prompt = PROMPTS["zero_shot"].format(question=question)
            else:
                additional_file = os.path.join(args.context_dir, f"{video_id}_master_dataset.jsonl")
                additional_data_str = load_additional_data(additional_file)
                final_prompt = PROMPTS[args.prompt_type].format(
                    question=question,
                    Additional=additional_data_str
                )

            # --- INFERENCE ---
            try:
                response = client.models.generate_content(
                    model="gemini-3-flash-preview", 
                    contents=[uploaded_video, final_prompt],
                    config={"response_mime_type": "application/json"}
                )

                ai_result = json.loads(response.text)
                final_record = {
                    "id": video_id,
                    "question": question,
                    "gemini_output": ai_result
                }

                f_out.write(json.dumps(final_record) + "\n")
                f_out.flush()

                processed_tasks.add((video_id, question))
                print(f"✅ Success! Saved prediction for {video_id}.")
                
                time.sleep(2) # Buffer for rate limits

            except Exception as e:
                print(f"❌ Gemini API Error during generation: {e}")

    print(f"\n✅ All queries processed! Results saved to {output_file}")

if __name__ == "__main__":
    main()
