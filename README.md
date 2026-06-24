# LAMAR-2 at MedGenVidQA 2026: Visual Answer Localization in Medical Videos via Multimodal LLM and Context-Augmented Prompting
This repository contains the official code, prompts, and evaluation scripts for the **LAMAR-2** submission to the MedGenVidQA 2026 Shared Task (Task C).  —  #1 on leaderboard.
## System Overview
We frame visual answer localization as a multimodal fusion problem, integrating raw video, timestamped ASR transcripts, and VLM-generated scene descriptions into structured contextual blocks, enabling the model to cross-reference spoken commentary against observable physical events. We show that targeted guidance, which forces the model to treat audio transcripts as supplementary hints with observable visual movements.
## Approach
<p align="center">
  <img src="asset/pipelineVLM.png" width="400" height="600" alt="LAMAR-2 Pipeline Architecture">
</p>



##  Repository Structure
```text
├── asset/                 # Images and diagrams for the README
│   └── pipelineVLM.png    # LAMAR-2 pipeline architecture diagram
├── data/                  
│   ├── queries/           # JSON files with questions and video URLs (e.g., task_c_test.json)
│   ├── raw_videos/        # Downloaded MedGenVidQA mp4 files
│   └── predictions/       # Output JSON files containing the LLM-predicted timestamps
├── prompts/               # JSON templates for Zero-Shot, Strict, CoT, and Heuristic Loose
├── src/
│   ├── 00_download_videos.py    # Downloads dataset from JSON URLs
│   ├── 01_asr_pipeline.py       # Qwen3-ASR-1.7B word-level timestamp generation
│   ├── 02_scene_vlm.py          # PySceneDetect + Qwen3-VL-8B description generation
│   ├── 03_fusion.py             # Aligns transcripts with scene descriptions
│   ├── 04_predict_timestamps.py # Gemini-3-Flash LLM inference to predict start/end boundaries
│   └── 05_evaluate.py           # Calculates IoU thresholds (0.3, 0.5, 0.7) and mIoU
├── requirements.txt       # Python dependencies 
└── README.md              # Project documentation
```
## Setup
```bash
git clone [https://github.com/biodatlab/medgenvidqa-lamar.git](https://github.com/biodatlab/medgenvidqa-lamar.git)
cd medgenvidqa-lamar
conda create -n lamar_env python=3.10
conda activate lamar_env
pip install -r requirements.txt
```
## Usage
### 1. Data Preparation
Download the official test JSON file containing the queries and video links directly into the `data/queries/` directory:
```bash
# Download the test JSON file from Google Drive
gdown 1UukkM5ppCyFwhEpK6C7YzKfyKgonTo77 -O data/queries/task_c_test.json
```
Download all the corresponding .mp4 videos into the data/raw_videos/ directory:
```bash
# Download videos based on the URLs in the JSON file
python src/00_download_videos.py --json_path data/queries/task_c_test.json --output_dir data/raw_videos/
```
### 2. Feature Extraction
Generate the word-level ASR timestamps and the visual scene descriptions using Qwen3-ASR and PySceneDetect + Qwen3-VL:
```bash
python src/01_asr_pipeline.py --video_dir data/raw_videos/ --output_dir data/asr_outputs/
python src/02_scene_vlm.py --video_dir data/raw_videos/ --output_dir data/scene_outputs/
```
### 3. Multimodal Fusion
Align the transcripts with the visual scene descriptions. This step utilizes our sliding-window text alignment algorithm to map the contextual text back to the original ASR word boundaries, ensuring high temporal accuracy before passing the context to the LLM.
```bash
python src/03_fusion.py --asr_dir data/asr_outputs/ --scene_dir data/scene_outputs/ --output_dir data/fused_context/
```
### 4. Timestamp Prediction
Run the LLM inference to predict the precise start and end boundaries. You can specify the prompt template (e.g., zero_shot, strict, cot, or heuristic_loose) from the prompts/ directory.
```bash
python src/04_predict_timestamps.py \
    --context_dir data/fused_context/ \
    --prompt_type heuristic_loose \
    --output_dir data/predictions/
```
### 5. Evaluation
Evaluate the predicted timestamps against the ground truth. This script calculates the Mean Intersection over Union (mIoU) and the Success Rates at varying IoU thresholds (0.3, 0.5, and 0.7) to benchmark against the leaderboard.
```bash
python src/05_evaluate.py \
    --pred_dir data/predictions/ \
    --gt_path data/queries/task_c_test.json
```

## Citation
If you use the LAMAR-2 pipeline or our Context-Augmented Prompting templates in your research, please cite our BioNLP 2026 paper:
```bash
@inproceedings{sermsrisuwan2026lamar2,
  title={LAMAR-2 at MedGenVidQA 2026: Visual Answer Localization in Medical Videos via Multimodal LLM and Context-Augmented Prompting},
  author={Sermsrisuwan, Watcharitpol and [Co-Authors]},
  booktitle={Proceedings of the BioNLP Shared Task at ACL 2026},
  year={2026}
}
```
