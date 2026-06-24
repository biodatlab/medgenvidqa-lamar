import os
import json
import time
import argparse
from google import genai

# --- PROMPT TEMPLATES ---
PROMPTS = {
    "zero_shot": """
You are an expert medical consultationist. You will be given a question related to the medical field. The supplement you will be provided is a medical content related video.

**Your Task:**
Identify the start and end timestamps for the entire segment in the video that comprehensively answers the given question.
DO NOT output the text of the answer. Only output the timestamps.

**Output Format:**
You MUST output a valid JSON object ONLY. Use the following exact keys:
{{
  "answer_start": "MM:SS",
  "answer_end": "MM:SS",
  "answer_start_second": integer,
  "answer_end_second": integer
}}

Your question: {question}
""",

    "strict": """
You are an expert medical video analyst. Your task is to identify the exact, continuous video segment that answers a given medical question using the provided video, visual context, and audio transcripts.

**Core Directives:**
1. **Visuals > Audio:** Visual evidence is the absolute priority; audio is strictly for verification. Do not select segments where the action/anatomy is discussed but not visually demonstrated.
2. **Tight Surgical Boundaries:** Timestamps must strictly bound the active physical procedure. Exclude all introductions, verbal planning, and text slides. Start exactly when the real operation begins and end exactly when it finishes.
3. **Visual Hierarchy:** If actual surgical footage is unavailable, fallback to a physical demonstration. If that is also unavailable, fallback to a text-based explanation.

**Output Format:**
Output ONLY a valid JSON object. Do not include any conversational text, markdown formatting outside the JSON, or explanations outside the "reasoning" key.

{{
  "reasoning": "Brief explanation prioritizing why the visual context (supported by audio) answers the question.",
  "answer_start": "MM:SS",
  "answer_end": "MM:SS"
}}

**Inputs:**
**The Question:** {question}
**Context and Transcription:** {Additional}
""",

    "cot": """
You are an expert medical video analyst. Your task is to identify the exact, continuous video segment that answers a given medical question using the provided video, visual context, and audio transcripts.

**Core Directives:**
1. **Visuals > Audio:** Visual evidence is the absolute priority; audio is strictly for verification. Do not select segments where the action/anatomy is discussed but not visually demonstrated.
2. **Tight Surgical Boundaries:** Timestamps must strictly bound the active physical procedure. Exclude all introductions, verbal planning, and text slides. Start exactly when the real operation begins and end exactly when it finishes.
3. **Do Not Echo Context:** The provided context and transcripts are rough temporal guides, NOT the final answer. You must independently discover the micro-boundaries within them. Never blindly copy the timestamps or durations of the provided input chunks.
4. **Visual Hierarchy:** If actual surgical footage is unavailable, fallback to a physical demonstration. If that is also unavailable, fallback to a text-based explanation.
5. **Visual Anchoring (Mandatory):** You must explicitly describe the exact visual event that marks the start and end of the segment BEFORE outputting timestamps.

**Output Format:**
Output ONLY a valid JSON object. Do not include any conversational text, markdown formatting outside the JSON, or explanations outside the specified keys.

{{
  "visual_start_anchor": "Describe the exact visual frame where the answer physically begins (e.g., 'Scalpel makes first contact with skin').",
  "visual_end_anchor": "Describe the exact visual frame where the answer physically concludes (e.g., 'Suture is cut and tool is removed from frame').",
  "reasoning": "Brief explanation of how these visual anchors directly answer the question, ensuring the timestamps are tighter than the provided transcript chunks.",
  "answer_start": "MM:SS",
  "answer_end": "MM:SS"
}}

**Inputs:**
**The Question:** {question}
**Rough Segments (Context/Audio):** {Additional}
""",

    "heuristic_loose": """
Role: Expert Medical Video Analyst.
Task: Identify the exact video segment that answers the question.

Question: {question}
Reference Notes (Transcripts & Scenes): {Additional}

Instructions:
Watch the video. The Reference Notes are provided only as a background hint. You must determine the precise start and end timestamps purely by observing the physical procedure in the video footage.

Output ONLY a valid JSON object:
{{
  "first_physical_movement": "Briefly state the visual action that starts the segment.",
  "final_physical_movement": "Briefly state the visual action that ends the segment.",
  "answer_start": "MM:SS",
  "answer_end": "MM:SS"
}}
"""
}

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
    parser.add_argument("--query_file", type=str, required=True, help="Path to JSON file containing queries (e.g., task_c_test.json)")
    parser.add_argument("--video_dir", type=str, required=True, help="Path to raw mp4 videos")
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
