# Video Processing Pipeline

### Table of Contents

- [Video Methods Overview](#video-methods-overview)
  - [Video Captioning Methods](#video-captioning-methods)
  - [Video Generation Methods](#video-generation-methods)
- [Usage](#usage)
  - [Running the Pipeline](#running-the-pipeline)
  - [Configuration File](#configuration-file)
  - [Adding New Steps](#adding-new-steps)
- [Input and Output Formats](#input-and-output-formats)
- [Examples](#examples)
  - [Prepare Data](#prepare-data)
  - [Run the Pipeline](#run-the-pipeline-1)
  - [View Results](#view-results)

### Video Methods Overview

This section describes two main categories of video processing algorithms: **Video Captioning** and **Video Generation**. Below are the detailed descriptions of each method, including model and functionality.

**Video Captioning Methods**

<table>
  <thead>
    <tr>
      <th>Name</th>
      <th>Model HuggingFace Path</th>
      <th>Description</th>
      <th>Official Repository or Paper</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>QwenVLCaptioner</td>
      <td>Qwen/Qwen2-VL-7B-Instruct</td>
      <td>Generates video captions using the Qwen2-VL model, suitable for varying levels of video caption granularity.</td>
      <td><a href="https://github.com/QwenLM/Qwen2-VL">Code</a><br><a href="https://arxiv.org/pdf/2409.12191">Paper</a></td>
    </tr>
    <tr>
      <td>LLaVANeXTVideoCaptioner</td>
      <td>llava-hf/LLaVA-NeXT-Video-7B-hf</td>
      <td>Uses the dynamic resolution-based LLaVA-NeXT model to improve the generation of semantically complex video captions.</td>
      <td><a href="https://github.com/LLaVA-VL/LLaVA-NeXT">Code</a><br><a href="https://llava-vl.github.io/blog/2024-01-30-llava-next/">Blog</a></td>
    </tr>
    <tr>
      <td>LLaVAOVCaptioner</td>
      <td>llava-hf/llava-onevision-qwen2-7b-ov-hf</td>
      <td>Optimized for video processing using the OneVision version of the LLaVA model, trained on large datasets to improve visual information handling.</td>
      <td><a href="https://github.com/LLaVA-VL/LLaVA-NeXT">Code</a><br><a href="https://arxiv.org/abs/2408.03326">Paper</a></td>
    </tr>
  </tbody>
</table>

**Video Generation Methods**

<table>
  <thead>
    <tr>
      <th>Name</th>
      <th>Model HuggingFace Path</th>
      <th>Description</th>
      <th>Official Repository or Paper</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>FLUXGenerator</td>
      <td>black-forest-labs/FLUX.1-dev</td>
      <td>Generates high-resolution videos using Flux, with strong instruction-following capabilities.</td>
      <td><a href="https://github.com/black-forest-labs/flux">Code</a></td>
    </tr>
    <tr>
      <td>StableDiffusionGenerator</td>
      <td>stabilityai/stable-diffusion-3.5-large</td>
      <td>Generates high-quality, detailed videos based on Stable Diffusion 3, with fast inference.</td>
      <td><a href="https://github.com/Stability-AI/StableDiffusion">Code</a><br><a href="https://arxiv.org/abs/2112.10752">Paper</a></td>
    </tr>
    <tr>
      <td>AllegroGenerator</td>
      <td>rhymes-ai/Allegro</td>
      <td>Generates dynamic videos up to 6 seconds, with a 15 FPS frame rate and 720p resolution, based on simple text input.</td>
      <td><a href="https://github.com/rhymes-ai/Allegro">Code</a><br><a href="https://arxiv.org/abs/2410.15458">Paper</a></td>
    </tr>
    <tr>
      <td>AnimateDiffGenerator</td>
      <td>guoyww/animatediff-motion-adapter-v1-5-2</td>
      <td>A framework based on Stable Diffusion, DreamBooth, and LoRA for generating animations by embedding animation capabilities into text-to-image models.</td>
      <td><a href="https://github.com/guoyww/AnimateDiff">Code</a><br><a href="https://arxiv.org/abs/2307.04725">Paper</a></td>
    </tr>
    <tr>
      <td>CogVideoXT2VGenerator</td>
      <td>rhymes-ai/Allegro</td>
      <td>A diffusion transformer-based model for text-to-video generation, excelling in motion representation, video duration, and text-video alignment.</td>
      <td><a href="https://github.com/THUDM/CogVideo">Code</a><br><a href="https://arxiv.org/abs/2408.06072">Paper</a></td>
    </tr>
    <tr>
      <td>CogVideoXGenerator</td>
      <td>THUDM/CogVideoX-5b-I2V</td>
      <td>A diffusion transformer-based image-to-video generation model, excelling in motion, duration, and text-video alignment.</td>
      <td><a href="https://github.com/THUDM/CogVideo">Code</a><br><a href="https://arxiv.org/abs/2408.06072">Paper</a></td>
    </tr>
    <tr>
      <td>I2VGenXLGenerator</td>
      <td>ali-vilab/i2vgen-xl</td>
      <td>Optimized video generation method that transforms a single static image into a high-quality, realistic animated video with continuous motion.</td>
      <td><a href="https://github.com/ali-vilab/VGen">Code</a><br><a href="https://arxiv.org/abs/2311.04145">Paper</a></td>
    </tr>
    <tr>
      <td>ModelScopeT2VGenerator</td>
      <td>damo-vilab/text-to-video-ms-1.7b</td>
      <td>A Stable Diffusion-based text-to-video model that ensures coherent frame generation and motion transitions, adaptable to different frame rates.</td>
      <td><a href="https://github.com/ali-vilab/VGen">Code</a><br><a href="https://arxiv.org/abs/2308.06571">Paper</a></td>
    </tr>
    <tr>
      <td>SVDGenerator</td>
      <td>stabilityai/stable-video-diffusion-img2vid-xt</td>
      <td>A diffusion-based video generation model that excels in text-to-video, image-to-video, and multi-view tasks, producing high-quality and motion-rich videos.</td>
      <td><a href="https://github.com/Stability-AI/generative-models">Code</a><br><a href="https://arxiv.org/abs/2311.15127">Paper</a></td>
    </tr>
  </tbody>
</table>

### Usage

1. **Running the Pipeline**: 

The pipeline is initiated by running the script `run_pipeline.py`. This script executes the entire video processing pipeline based on the configuration file, which includes:

- **Preprocessing**: Converts input data into the required format for the pipeline.
- **Model Inference**: Runs the model steps as defined in the YAML configuration file.
- **Postprocessing**: Converts the output from model inference into the final required format.

**Command-line Execution**

```bash
python run_pipeline.py --config ./configs/VideoCaption.yaml
```

2. **Configuration File**: 

The configuration file is written in YAML format and defines the pipeline’s inputs, outputs, and each processing step.

**Example Configuration File**


```yaml
meta_path: ./data/video/test_video_captioner.jsonl # path for the meta data, the format is fixed for image and video
meta_folder: data/video # only for captioner
base_folder: video_intermediate_results/ # path to save the intermediate results
save_folder: results/video_captioner # path to save the caption
image_key: image # key for the image in the meta data
text_key: text # key for the text in the meta data
video_key: video # key for the video in the meta data

steps:
  - type: VideoCaptioner
    name: Qwen2VLCaptioner
    config:
      model_path: /mnt/hwfile/mllm/niujunbo/model/Qwen/Qwen2-VL-7B-Instruct
      trust_remote_code: true
      tensor_parallel_size: 1
      max_model_len: 2048
      max_num_seqs: 128
      temperature: 0.7
      top_p: 0.9
      top_k: 50
      repetition_penalty: 1.2
      prompt: "Please describe the video in detail."
      batch_size: 1000
```

### Configuration Parameters Explanation

- **meta_path**: Path to the metadata file in JSONL format, with one JSON object per line containing image/video information.
  
- **meta_folder**: Folder containing images or videos.
  
- **base_folder**: Root directory to store intermediate results.
  
- **save_folder**: Folder where the final results are saved.
  
- **image_key**: The key in metadata representing the image path.
  
- **text_key**: The key in metadata representing the text content.
  
- **video_key**: The key in metadata representing the video path.
  
- **steps**: Defines the processing steps to be executed. Each step contains a type, name, and specific configuration.


3. **Adding New Steps**: 

To add a new video processing step, simply add the configuration under the `steps` section. Each step can define either a **Video Captioning** or **Video Generation** method. You can also specify additional parameters required by each model.

### Input and Output Formats

<!-- 1. **Input**:
   - Video file paths
   - Metadata for video processing
   - Image-text pairs (for captioning)

2. **Output**:
   - Video captions (as text files)
   - Generated video files (in formats like MP4 or GIF)
   - Intermediate results for inspection -->
1. **Input Format**

The pipeline supports various input formats, including CSV, TSV, Parquet, JSON, and JSONL. The framework will convert different input formats into a unified dictionary for storage, and the intermediate output format is JSONL. Below is an explanation based on the JSONL format.

For the input JSONL file, each line contains a JSON object describing a single data entry. Depending on the processing type, the JSON object should include the corresponding keys.

**JSONL Format for Video Captioning**

```json
{"id": "000000000", "video": "demo/1.mp4"}

{"id": "000000001", "video": "demo/2.mp4"}

{"id": "000000002", "video": "demo/3.mp4"}
```

You're correct! In Markdown, the proper way to bold text is to surround the text with double asterisks (`**`) or double underscores (`__`). However, when you're using an asterisk in front of an item (e.g., `• **image**`), Markdown can interpret it as part of the list formatting, causing the bold formatting not to work as expected. 

A better approach would be to add the bolding directly after the list marker without breaking the formatting. Here's the corrected version:



### Key Fields Explanation

- **image**: The name of the image file.
  
- **video**: The name of the video file.
  
- **text**: The text description or prompt related to the image or video.

This way, the bold formatting will work correctly for each item in the list.

For the captioning methods, the format is similar; you simply need to specify the image/video name for which the description is to be generated.

### Examples

1. **Prepare Data**

Ensure that your video files and metadata are organized correctly. For captioning, a JSONL file containing video file paths, captions, and additional information is expected.

**Example Metadata File (`test_video_captioner.jsonl`)**

```json
{"id": "000000000", "video": "demo/1.mp4"}

{"id": "000000001", "video": "demo/2.mp4"}

{"id": "000000002", "video": "demo/3.mp4"}
```

2. **Run the Pipeline**

```bash
python run_pipeline.py --config ./configs/VideoCaption.yaml
```

3. **View Results**

After running the pipeline, inspect the `results/video_captioner` directory for the generated captions and videos.

### Project Structure

```
Dataflow-Gen/
├── data/
│  ├── image/
│  │  ├── test_image_captioner.jsonl
│  │  ├── videoss/
│  │    ├── cat.jpg
│  │    ├── sunset.png
│  ├── video/
│  │  ├── test_video_captioner.jsonl
│  │  ├── videos/
│  │    ├── cat.mp4
│  │    ├── dog.mp4
├── intermediate_results/
├── results/
├── src/
│  ├── config.py
│  ├── pipeline/
│  │  ├── manager.py
│  │  ├── steps.py
│  │  ├── wrappers.py
│  ├── data/
│    ├── DataManager.py
│    ├── Dataset.py
│  ├── utils/
│    ├── data_utils.py
│    ├── registry.py
├── run_pipeline.py
├── requirements.txt
├── config.yaml
├── README.md
```
