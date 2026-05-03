# LAMAR-2 at MedGenVidQA 2026: Visual Answer Localization in Medical Videos via Multimodal LLM and Context-Augmented Prompting
This repository contains the official code, prompts, and evaluation scripts for the **LAMAR-2** submission to the MedGenVidQA 2026 Shared Task (Task C).  —  #1 on leaderboard.
## System Overview
We frame visual answer localization as a multimodal fusion problem, integrating raw video, timestamped ASR transcripts, and VLM-generated scene descriptions into structured contextual blocks, enabling the model to cross-reference spoken commentary against observable physical events. We show that targeted guidance, which forces the model to treat audio transcripts as supplementary hints with observable visual movements.
![LAMAR-2 Pipeline Architecture](column_framework.png)

##  Repository Structure
```text
├── data/                  # MedGenVidQA video data and annotations
├── prompts/               # JSON templates for Zero-Shot, Strict, CoT, and Heuristic Loose
├── src/
│   ├── 01_asr_pipeline.py # Qwen3-ASR-1.7B word-level timestamp generation
│   ├── 02_scene_vlm.py    # PySceneDetect + Qwen3-VL-8B description generation
│   ├── 03_fusion.py       # Aligns transcripts with scene descriptions
│   └── 04_evaluate.py     # Gemini-3-Flash inference and IoU calculation
├── requirements.txt       # Python dependencies 
├── column_framework.png   # Pipeline architecture diagram
└── README.md
