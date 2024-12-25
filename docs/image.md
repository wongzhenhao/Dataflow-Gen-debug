# Image Processing Pipeline

### Table of Contents

- [Image Methods Overview](#image-methods-overview)
  - [Image Captioning Methods](#image-captioning-methods)
  - [Image Generation Methods](#image-generation-methods)
- [Usage](#usage)
  - [Running the Pipeline](#running-the-pipeline)
  - [Configuration File](#configuration-file)
  - [Adding New Steps](#adding-new-steps)
- [Input and Output Formats](#input-and-output-formats)
- [Examples](#examples)
  - [Prepare Data](#prepare-data)
  - [Run the Pipeline](#run-the-pipeline-1)
  - [View Results](#view-results)

### Image Methods Overview

This section covers two main categories of image processing algorithms: **Image Captioning** and **Image Generation**. Below are the detailed descriptions of each method, including model and functionality.

**Image Captioning Methods**

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
      <td>BLIPCaptioner</td>
      <td>Salesforce/blip2-opt-2.7b</td>
      <td>Generates image captions using the BLIP2 model based on QFormer, widely used for datasets like LLaVA Pretrain 558.</td>
      <td><a href="https://github.com/salesforce/LAVIS/tree/main/projects/blip2">Code</a><br><a href="https://arxiv.org/abs/2301.12597">Paper</a></td>
    </tr>
    <tr>
      <td>LLaVACaptioner</td>
      <td>llava-hf/llava-1.5-7b-hf</td>
      <td>Generates image captions using the LLaVA model, suitable for simpler tasks.</td>
      <td><a href="https://github.com/haotian-liu/LLaVA">Code</a><br><a href="https://arxiv.org/abs/2304.08414">Paper</a></td>
    </tr>
    <tr>
      <td>LLaVANeXTCaptioner</td>
      <td>llava-hf/llava-v1.6-mistral-7b-hf</td>
      <td>Uses LLaVA-NeXT with dynamic resolution, improving the ability to generate semantically complex image captions.</td>
      <td><a href="https://github.com/LLaVA-VL/LLaVA-NeXT">Code</a><br><a href="https://llava-vl.github.io/blog/2024-01-30-llava-next/">Blog</a></td>
    </tr>
    <tr>
      <td>LLaVAOneVisionCaptioner</td>
      <td>llava-hf/llava-onevision-qwen2-7b-ov-hf</td>
      <td>Optimized for vision processing using the OneVision version of the LLaVA model, trained on large datasets.</td>
      <td><a href="https://github.com/LLaVA-VL/LLaVA-NeXT">Code</a><br><a href="https://arxiv.org/abs/2408.03326">Paper</a></td>
    </tr>
    <tr>
      <td>MLLamaCaptioner</td>
      <td>meta-llama/Llama-3.2-11B-Vision-Instruct</td>
      <td>Generates image captions using Meta's LLaMA Vision model, supporting high-parallelism captioning.</td>
      <td><a href="https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/">Official Site</a></td>
    </tr>
    <tr>
      <td>Phi3VCaptioner</td>
      <td>microsoft/Phi-3-vision-128k-instruct</td>
      <td>Based on Microsoft's Phi-3 model, specializing in high-quality instruction-based image captioning.</td>
      <td><a href="https://github.com/microsoft/Phi-3CookBook">Code</a><br><a href="https://azure.microsoft.com/en-us/blog/introducing-phi-3-redefining-whats-possible-with-slms/">Official Site</a><br><a href="https://arxiv.org/abs/2404.14219">Paper</a></td>
    </tr>
    <tr>
      <td>QwenVLCaptioner</td>
      <td>Qwen/Qwen2-VL-7B-Instruct</td>
      <td>Generates image captions based on the Qwen2 model, supporting multi-turn conversational captioning.</td>
      <td><a href="https://github.com/QwenLM/Qwen2.5">Code</a><br><a href="https://arxiv.org/abs/2407.10671">Paper</a></td>
    </tr>
  </tbody>
</table>


**Image Generation Methods**

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
      <td>Generates high-resolution images based on the Flux implementation, with strong instruction-following capabilities.</td>
      <td><a href="https://github.com/black-forest-labs/flux">Code</a></td>
    </tr>
    <tr>
      <td>StableDiffusionGenerator</td>
      <td>stabilityai/stable-diffusion-3.5-large</td>
      <td>Generates high-quality, detailed images based on Stable Diffusion 3, with fast inference.</td>
      <td><a href="https://github.com/Stability-AI/StableDiffusion">Code</a><br><a href="https://arxiv.org/abs/2112.10752">Paper</a></td>
    </tr>
  </tbody>
</table>


### Usage

1. **Running the Pipeline**: 

The pipeline is initiated by running the script `run_pipeline.py`. This script executes the entire image processing pipeline based on the configuration file, which includes:

- **Preprocessing**: Converts input data into the format required by the pipeline.
- **Model Inference**: Runs the model steps as defined in the YAML configuration file.
- **Postprocessing**: Converts the output from model inference into the final required format.

**Command-line Execution**

```bash
python run_pipeline.py --config path/to/config.yaml
```

2. **Configuration File**: 

The configuration file is written in YAML format and defines the pipeline’s inputs, outputs, and each processing step.

**Example Configuration File**

```yaml
meta_path: data/image/test_image_captioner.jsonl # Path to metadata file, with images and videos in the correct format
meta_folder: data/image # Only for caption generation
base_folder: intermediate_results/ # Folder to save intermediate results
save_folder: results/ # Folder to save generated images
image_key: image # Key in metadata representing the image
text_key: text # Key in metadata representing the text
video_key: video # Key in metadata representing the video

steps:
 - type: ImageCaptioner
   name: LLaVACaptioner
   config: 
     model_path: /mnt/petrelfs/liuzheng/cjz/llava-1.5-7b-hf
     trust_remote_code: true
     tensor_parallel_size: 2
     max_model_len: 2048
     max_num_seqs: 128
     temperature: 0.7
     top_p: 0.9
     top_k: 50
     repetition_penalty: 1.2
     prompt: What is the content of this image?
     repeat_time: 5
     batch_size: 1000
 - type: ImageGenerator
   name: FLUXGenerator
   config:
     model_path: black-forest-labs/FLUX.1-dev
     height: 1024
     width: 1024
     guidance_scale: 4.5
     num_inference_steps: 50
     max_sequence_length: 512
     batch_size: 1
     device: cuda
 - type: other
```

**Configuration Parameters Explanation**

	•	**meta_path**: Path to the metadata file in JSONL format, with one JSON object per line containing image/video information.
	
	•	**meta_folder**: Folder containing images or videos.
	
	•	**base_folder**: Root directory to store intermediate results.
	
	•	**save_folder**: Folder where the final results are saved.
	
	•	**image_key**: The key in metadata representing the image path.
	
	•	**text_key**: The key in metadata representing the text content.
	
	•	**video_key**: The key in metadata representing the video path
	
	•	**steps**: Defines the processing steps to be executed. Each step contains a type, name, and specific configuration.

3. **Adding New Steps**

To add a new processing step, simply add the corresponding configuration in the `steps` list. For example, to add a new text generation step:

```yaml
- type: TextGenerator
 name: GPTTextGenerator
 config:
  model: gpt-4o
  api_key: your_api_key_here
  max_num_seqs: 512
  temperature: 0.2
  sys_prompt: You are an excellent assistant.
  prompt: Generate a summary for the following content.
  batch_size: 1
```

### Input and Output Formats

1. **Input Format**

The pipeline supports various input formats, including CSV, TSV, Parquet, JSON, and JSONL. The framework will convert different input formats into a unified dictionary for storage, and the intermediate output format is JSONL. Below is an explanation based on the JSONL format.

For the input JSONL file, each line contains a JSON object describing a single data entry. Depending on the processing type, the JSON object should include the corresponding keys.

**JSONL Format for Image Captioning**

{"image": "image1.jpg", "text": "A cat sitting on a windowsill."}

{"image": "image2.png", "text": "A beautiful sunset over the mountains."}

{"video": "video1.mp4", "text": "A timelapse of city traffic at night."}

**Key Fields Explanation**

	•	**image**: The name of the image file.
	
	•	**video**: The name of the video file.
	
	•	**text**: The text description or prompt related to the image or video.

For the captioning methods, the format is similar; you simply need to specify the image/video name for which the description is to be generated.

2. **Output Format**

The output format depends on the steps executed. Typically, results are saved in the folder specified by save_folder in the configuration file, and they are organized by step name.

For caption generation methods, the generated results are saved in the JSONL file under the user-specified key (e.g., ‘text’), while for image generation methods, the images/videos are stored in the specified directory.

3.  **Intermediate Results**

Intermediate results are stored in the base_folder. Each step creates a subfolder under this directory to store its outputs.

Example directory structure for intermediate results:
```
intermediate_results/
├── step_0_preprocess_format/
│  ├── result.jsonl
├── step_1_ImageCaptioner_LLaVACaptioner/
│  ├── result.jsonl
├── step_2_ImageGenerator_FLUXGenerator/
│  ├── generated_images/
│    ├── image1_generated.jpg
│    ├── image2_generated.png
```
4. **Final Results**

The final results are stored in the save_folder. The content depends on the output of the last step. For example, if the last step is image generation, the final results will include the generated image files.

Example directory structure for final results:
```
results/
├── generated_images/
│  ├── image1_generated.jpg
│  ├── image2_generated.png
```
### Examples

1. **Prepare Data**

Create a JSONL file with image captions, for example, test_image_captioner.jsonl:

{"image": "images/cat.jpg", "text": "A cat sitting on a windowsill."}

{"image": "images/sunset.png", "text": "A beautiful sunset over the mountains."}

2. **Run the Pipeline**

Use the provided sample configuration file config.yaml to run the pipeline:
```bash
python run_pipeline.py --config config.yaml
```
3. **View Results**

The generated images and captions will be saved in the intermediate_results/ and results/ directories.

### Project Structure
```
Dataflow-Gen/
├── data/
│  ├── image/
│  │  ├── test_image_captioner.jsonl
│  │  ├── images/
│  │    ├── cat.jpg
│  │    ├── sunset.png
│  ├── video/
│    ├── video1.mp4
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
