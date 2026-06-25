# AI Content Generator System

### A Multi-Agent Framework for Generic Text, Image, Audio, and Video Generation

**Institution:** SRH University of Applied Sciences, Heidelberg
**Program:** M.Sc. Applied Data Science and Artificial Intelligence
**Course:** Case Studies 1
**Status:** Text and Image modules complete; Audio and Video modules under active development

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [System Architecture](#2-system-architecture)
3. [Data Flow](#3-data-flow)
4. [Technology Stack](#4-technology-stack)
5. [AI Pipeline](#5-ai-pipeline)
6. [Project Structure](#6-project-structure)
7. [Installation and Setup](#7-installation-and-setup)
8. [Usage](#8-usage)
9. [Evaluation and Benchmarking](#9-evaluation-and-benchmarking)
10. [Deployment](#10-deployment)
11. [Current Limitations and Future Work](#11-current-limitations-and-future-work)
12. [References](#12-references)

---

## 1. Introduction

The rapid evolution of generative artificial intelligence has created a strong demand for systems that can produce high-quality content across multiple modalities. Most existing solutions operate as single-step black-box generators, producing outputs without any internal mechanism for validation, refinement, or quality control. This often results in inconsistencies, hallucinations, and outputs misaligned with user intent.

The AI Content Generator System addresses these limitations through a structured multi-agent framework that decomposes the generation process into five distinct stages: analysis, retrieval, generation, optimization, and evaluation. Each stage is implemented as a dedicated agent, allowing modular development, independent testing, and stage-specific improvements.

The first phase of the project delivered fully functional text and image generation modules with comprehensive benchmarking. The current phase extends the framework to include audio and video generation, applying the same architectural principles while introducing modality-specific evaluation strategies such as CLAP-based audio scoring and temporal CLIP-based video coherence analysis.

### 1.1 Objectives

- Design a unified, extensible architecture capable of supporting multiple content modalities.
- Implement iterative quality refinement loops with threshold-based retry logic.
- Provide transparent, auditable evaluation through structured metadata tracking.
- Extend the current text and image system with audio and video generation pipelines.
- Maintain consistency in workflow orchestration across all modalities.

### 1.2 Key Contributions

- A LangGraph-based stateful workflow that orchestrates multi-agent pipelines.
- A retrieval-augmented generation (RAG) component for contextual grounding of text outputs.
- A dual-metric evaluation framework combining semantic alignment and aesthetic quality for images.
- A planned extension to audio and video with CLAP and temporal coherence metrics respectively.
- A containerized deployment supporting reproducibility across environments.

---

## 2. System Architecture

The system follows a layered architecture in which each layer encapsulates a specific concern. Communication between layers occurs through well-defined interfaces, ensuring separation of responsibilities and maintainability.

### 2.1 Architectural Layers

| Layer | Responsibility |
|-------|----------------|
| User Interaction Layer | Provides the Streamlit-based web interface for prompt input, parameter configuration, and output review. |
| Application Layer | Hosts the Supervisor and Router components that coordinate workflow execution and direct requests to the appropriate generation agent. |
| Content Generation Layer | Contains the modality-specific agents (Text, Image, Audio, Video), each implementing its own pipeline of analyzer, generator, optimizer, reviewer, and exporter nodes. |
| Shared Services Layer | Provides cross-cutting functionality including API configuration, execution logging, monitoring, and quality evaluation services. |
| Data and Persistence Layer | Manages the retrieval knowledge base, generated outputs, prompt history, calibration datasets, and benchmark results. |

### 2.2 Component Interaction

```
┌─────────────────────────────────────────────────────────────┐
│                  User Interaction Layer                     │
│              (Streamlit Web Interface)                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Application Layer                         │
│        Supervisor (Workflow Coordinator)  →  Router         │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ Text Agent   │      │ Image Agent  │      │ Audio Agent  │
│              │      │              │      │  (Planned)   │
└──────────────┘      └──────────────┘      └──────────────┘
                                                    │
                                            ┌──────────────┐
                                            │ Video Agent  │
                                            │  (Planned)   │
                                            └──────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                Shared Services Layer                        │
│   API Config  │  Logging  │  Evaluation (CLIP, CLAP)        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Data and Persistence Layer                     │
│  Knowledge Base │ Outputs │ Metadata │ Calibration Data     │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Data Flow

The data flow describes how a user request is transformed into a validated output through the system's pipeline. Each agent receives a shared state object containing the prompt, intermediate results, evaluation scores, and metadata, which is progressively enriched at each stage.

### 3.1 End-to-End Flow

```
User Input
    │
    ▼
[1] Web Interface captures prompt, mode, and parameters
    │
    ▼
[2] Supervisor initializes the shared state object
    │
    ▼
[3] Router selects the appropriate agent (text / image / audio / video)
    │
    ▼
[4] Analyzer extracts intent, attributes, and constraints
    │
    ▼
[5] Retriever or Prompt Enhancer enriches the input
    │
    ▼
[6] Generator produces the initial output
    │
    ▼
[7] Optimizer refines the output based on evaluation feedback
    │
    ▼
[8] Reviewer scores the output against quality thresholds
    │
    ├── Score below threshold ──► Return to Optimizer (retry loop)
    │
    └── Score meets threshold ──► Exporter saves output and metadata
                                                │
                                                ▼
                                       Output returned to user
```

### 3.2 State Object

The shared state object carries the following fields across the pipeline:

| Field | Description |
|-------|-------------|
| `prompt` | Original user input |
| `enhanced_prompt` | Refined prompt after analysis and enhancement |
| `retrieved_context` | Optional context fetched from the knowledge base |
| `generated_output` | Raw output produced by the generator |
| `optimized_output` | Refined output after the optimization stage |
| `evaluation_scores` | Quantitative scores from the reviewer |
| `metadata` | Model configuration, timestamps, retry counts, and decisions |
| `status` | Current pipeline stage and pass/fail state |

---

## 4. Technology Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| Workflow Orchestration | LangGraph | Stateful directed-graph workflow execution |
| Web Interface | Streamlit | Interactive user interface |
| Language Models | GPT-4o, LLaMA 3.3-70B, Kimi K2, Qwen3-32B | Text generation backends |
| Image Models | OpenAI Image-1 Mini, Stability SDXL, Freepik Mystic | Image generation backends |
| Audio Models (Planned) | ElevenLabs, Google Cloud TTS, Tortoise TTS | Text-to-speech generation |
| Video Models (Planned) | Runway Gen-3, Pika 2.0, Stable Video Diffusion | Text-to-video generation |
| Retrieval | TF-IDF + cosine similarity | Contextual document retrieval |
| Image Evaluation | OpenCLIP | Semantic alignment between prompt and image |
| Image Evaluation | Aesthetic Scorer (regression on CLIP embeddings) | Visual quality assessment |
| Audio Evaluation (Planned) | CLAP | Semantic alignment between prompt and audio |
| Video Evaluation (Planned) | Temporal CLIP, optical flow | Frame coherence and motion smoothness |
| Containerization | Docker, Docker Compose | Portable deployment |
| Container Management | Portainer | Institutional Data Lab deployment |
| Programming Language | Python 3.10+ | Primary implementation language |
| Configuration | YAML, .env | Model selection and credentials |
| Logging | Structured JSON logs | Auditability and benchmarking |

---

## 5. AI Pipeline

The AI pipeline is the conceptual backbone of the system. Every modality implements the same five-stage pattern, with stage-specific logic tailored to the content type. This uniformity is what allows the system to remain extensible as new modalities are added.

### 5.1 Common Pipeline Pattern

```
   ┌───────────┐      ┌───────────┐      ┌───────────┐
   │ Analyzer  │ ───► │ Retriever │ ───► │ Generator │
   └───────────┘      └───────────┘      └───────────┘
                                                │
                                                ▼
   ┌───────────┐      ┌───────────┐      ┌───────────┐
   │ Exporter  │ ◄─── │ Reviewer  │ ◄─── │ Optimizer │
   └───────────┘      └───────────┘      └───────────┘
                            │                   ▲
                            └───────────────────┘
                            (retry loop on failure)
```

### 5.2 Text Generation Pipeline

The text pipeline begins with prompt analysis to extract intent, tone, keywords, and content type. The retriever then queries a TF-IDF indexed knowledge base, returning the top-k relevant documents which are combined with the original prompt. The generator produces text via a configured LLM. The optimizer applies rule-based refinement focused on readability, sentiment alignment, and repetition reduction. The reviewer computes a composite quality score; if it falls below the threshold, the workflow returns to the optimizer for further iteration.

### 5.3 Image Generation Pipeline

The image pipeline analyzes the prompt for visual attributes and composition requirements. A prompt enhancer enriches the input with stylistic and artistic directives. The generator produces images via a diffusion-based model. The reviewer evaluates outputs using two metrics: CLIP-based semantic similarity for prompt-image alignment, and an aesthetic regression model for visual quality. Failed evaluations trigger regeneration with adjusted prompts.

### 5.4 Audio Generation Pipeline (In Development)

The audio pipeline mirrors the text and image pattern. The analyzer extracts speech characteristics such as tone, pace, and emotional register. A prompt enhancer adds voice directives and pronunciation hints. The generator invokes a text-to-speech model. The optimizer adjusts prosodic features such as pitch, speed, and emphasis. The reviewer evaluates outputs using CLAP for semantic alignment, spectral analysis for clarity, and prosodic consistency metrics for naturalness.

### 5.5 Video Generation Pipeline (In Development)

The video pipeline introduces a scene analyzer that extracts narrative flow, pacing, and visual style. A prompt builder combines textual content with visual directives. The generator invokes a text-to-video model. A frame optimizer ensures coherence across frames. The reviewer computes temporal CLIP scores frame by frame, analyzes optical flow for motion smoothness, and detects flicker artifacts. An optional compositor synchronizes generated audio with video if both modalities are used together.

### 5.6 Threshold-Based Retry Logic

All pipelines share a uniform retry mechanism. If the reviewer's score falls below the configured threshold, the workflow returns to the optimizer up to a maximum number of attempts. This ensures consistent quality while preventing infinite loops on inherently difficult prompts.

---

## 6. Project Structure

```
ai-content-generator/
│
├── app.py                          Streamlit application entry point
├── config.yaml                     Model selection and threshold configuration
├── requirements.txt                Python dependencies
├── Dockerfile                      Container image definition
├── docker-compose.yml              Multi-container orchestration
├── .env.example                    Template for environment variables
│
├── src/
│   ├── generator.py                Top-level ContentGenerator class
│   │
│   ├── agents/
│   │   ├── text_agent.py           Text generation workflow
│   │   ├── image_agent.py          Image generation workflow
│   │   ├── audio_agent.py          Audio generation workflow (planned)
│   │   └── video_agent.py          Video generation workflow (planned)
│   │
│   ├── evaluators/
│   │   ├── text_evaluator.py       Readability, sentiment, repetition metrics
│   │   ├── image_evaluator.py      CLIP and aesthetic scoring
│   │   ├── audio_evaluator.py      CLAP and prosody metrics (planned)
│   │   └── video_evaluator.py      Temporal CLIP and optical flow (planned)
│   │
│   ├── retrievers/
│   │   ├── base_retriever.py       Abstract retriever interface
│   │   └── tfidf_retriever.py      TF-IDF based retrieval
│   │
│   ├── optimizers/
│   │   ├── text_optimizer.py       Rule-based text refinement
│   │   ├── audio_optimizer.py      Prosody adjustment (planned)
│   │   └── video_optimizer.py      Frame coherence (planned)
│   │
│   ├── orchestration/
│   │   ├── supervisor.py           Workflow coordinator
│   │   └── router.py               Modality routing logic
│   │
│   └── utils/
│       ├── logging.py              Structured logging
│       └── helpers.py              Common utilities
│
├── tests/                          Unit and integration tests
├── benchmarks/                     Benchmark scripts and results
├── notebooks/                      Exploratory analysis
├── docs/                           Extended documentation
└── examples/                       Usage examples per modality
```

---

## 7. Installation and Setup

### 7.1 Prerequisites

| Requirement | Specification |
|-------------|---------------|
| Python | 3.10 or higher |
| CUDA | 12.0 or higher (for GPU-accelerated generation) |
| GPU | NVIDIA RTX 3090 or higher; A100/H100 recommended for video |
| RAM | 32 GB minimum; 64 GB recommended for video |
| Docker | Latest stable version (for containerized deployment) |

### 7.2 Local Installation

```bash
git clone <repository-url>
cd ai-content-generator
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

The `.env` file must contain valid API keys for the model providers being used.

### 7.3 Docker Installation

```bash
docker-compose up -d
```

The application becomes available at `http://localhost:8501`.

---

## 8. Usage

### 8.1 Web Interface

Launch the Streamlit application:

```bash
streamlit run app.py
```

The interface allows users to select the generation mode, configure parameters such as tone, style, and language, submit prompts, and view generated outputs along with their associated metadata.

### 8.2 Programmatic Interface

```python
from ai_content_generator import ContentGenerator, GenerationMode

generator = ContentGenerator(model_config="gpt-4o")

result = generator.generate(
    prompt="Write a blog post about AI trends",
    mode=GenerationMode.TEXT,
    tone="professional",
    language="English"
)
```

---

## 9. Evaluation and Benchmarking

### 9.1 Evaluation Metrics by Modality

| Modality | Primary Metrics | Secondary Metrics |
|----------|-----------------|-------------------|
| Text | Composite quality score, readability, sentiment alignment | Repetition penalty, keyword coverage |
| Image | CLIP similarity, aesthetic score | Color balance, composition |
| Audio (Planned) | CLAP similarity, clarity, prosody consistency | Loudness, frequency balance |
| Video (Planned) | Temporal CLIP, optical flow smoothness, aesthetic | Scene consistency, flicker |

### 9.2 Text Generation Benchmark Results

| Model | Optimized Score | Time (sec) | Improvement | Revision Rate |
|-------|-----------------|------------|-------------|---------------|
| Qwen3-32B | 71.0 | 6.2 | +26.2 | 50% |
| GPT-4o | 74.9 | 22.1 | +27.6 | 80% |
| LLaMA 3.3-70B | 76.2 | 8.1 | +31.8 | 70% |
| Kimi K2 | 76.4 | 6.8 | +33.6 | 85% |

### 9.3 Image Generation Benchmark Results

| Model | Pass Rate | CLIP Score | Aesthetic Score | Retry Rate |
|-------|-----------|------------|-----------------|------------|
| OpenAI Image-1 Mini | 100% | 0.330 | 6.86 | 0% |
| Stability SDXL | 80% | 0.342 | 6.49 | 20% |
| Freepik Mystic | 80% | 0.326 | 6.77 | 30% |

### 9.4 Key Findings

The benchmarking experiments confirmed that the optimization stage consistently improves output quality across all evaluated text models, with average improvements between 26 and 34 points. The results also demonstrated a quality-latency trade-off: larger models produce higher-quality outputs at the cost of significantly increased computation time. For image generation, the combination of CLIP and aesthetic scoring proved effective for filtering low-quality outputs, with pass rates ranging from 80% to 100% depending on the model.

---

## 10. Deployment

The system was containerized using Docker and deployed on an institutional Data Lab server through Portainer. This deployment strategy provides portability across environments, reproducibility of behavior, scalability through container replication, and simplified maintenance through container replacement.

Cloud deployment is supported on AWS, Google Cloud Platform, and Azure. The container image can be pushed to the respective container registry and deployed using each platform's managed container services.

---

## 11. Current Limitations and Future Work

### 11.1 Current Limitations

- Audio and video generation pipelines are under development and not yet production-ready.
- The TF-IDF retriever does not capture semantic similarity beyond lexical overlap.
- Optimization is currently rule-based and does not adapt based on past outcomes.
- The system does not yet support cross-modal generation (for example, generating video from text and audio together).

### 11.2 Future Work

| Phase | Focus Area |
|-------|------------|
| Phase 1 | Complete audio generation with CLAP-based evaluation and prosody refinement |
| Phase 2 | Complete video generation with temporal CLIP and optical flow analysis |
| Phase 3 | Implement multimodal synchronization between text, audio, and video |
| Phase 4 | Replace TF-IDF with semantic retrieval using FAISS or Pinecone |
| Phase 5 | Introduce learning-based optimization to replace rule-based refinement |
| Phase 6 | Add personalization based on user preferences and history |
| Phase 7 | Enable real-time interactive feedback loops for output refinement |

---

## 12. References

1. Vaswani, A., et al. "Attention Is All You Need." *Advances in Neural Information Processing Systems*, 2017.
2. Radford, A., et al. "Learning Transferable Visual Models From Natural Language Supervision." *Proceedings of ICML*, 2021.
3. Lewis, P., et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *NeurIPS*, 2020.
4. Ho, J., et al. "Denoising Diffusion Probabilistic Models." *NeurIPS*, 2020.
5. Salton, G., and Buckley, C. "Term-weighting Approaches in Automatic Text Retrieval." *Information Processing & Management*, vol. 24, no. 5, 1988.
6. Hutto, C., and Gilbert, E. "VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text." *ICWSM*, 2014.
7. LangChain. "LangGraph Documentation," 2025.
8. OpenCLIP. "OpenCLIP: Open Source CLIP Implementation," 2023.

---

## Project Team

| Name | Role |
|------|------|
| Faezeh Khosravi | Developer |
| Madhumitha Ramakrishnan | Developer |
| Saravanakumar Andamuthuvallal | Developer |
| Shreya Shetty | Developer |

**Institution:** SRH University of Applied Sciences, Heidelberg
**Course:** Case Studies 1
**Program:** M.Sc. Applied Data Science and Artificial Intelligence
