# Image Processing Pipeline

## Table of Contents

- [Usage](#1-usage)
- [Input and Output Formats](#2-input-and-output-formats)
- [Examples](#3-examples)

## 1 Usage

### 1.1 Running the Pipeline

The pipeline is initiated by running the script `run_pipeline.py`. This script executes the entire image processing pipeline based on the configuration file, which includes:

- **Preprocessing**: Converts input data into the format required by the pipeline.
- **Model Inference**: Runs the model steps as defined in the YAML configuration file.
- **Postprocessing**: Converts the output from model inference into the final required format.

### 1.2 Command-line Execution

```bash
python run_pipeline.py --config configs/ImageCaption.yaml
```

### 1.3 Configuration File

The configuration file is written in YAML format and defines the pipeline’s inputs, outputs, and each processing step.

**Example Configuration File**

path: configs/ImageCaption.yaml

```yaml
# example yaml file

meta_path: data/image/test_image_captioner.jsonl # path for the meta data, the format is fixed for image and video
meta_folder: data/image # only for captioner
base_folder: image_intermediate_results/ # path to save the intermediate results
save_folder: results/image_captioner # path to save the generated images
image_key: image # key for the image in the meta data
text_key: text # key for the text in the meta data
video_key: video # key for the video in the meta data

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
```

**Configuration Parameters Explanation**

**meta_path**: Path to the metadata file in JSONL format, with one JSON object per line containing image/video information.
	
**meta_folder**: Folder containing images or videos.
	
**base_folder**: Root directory to store intermediate results.
	
**save_folder**: Folder where the final results are saved.
	
**image_key**: The key in metadata representing the image path.
	
**text_key**: The key in metadata representing the text content.
	
**video_key**: The key in metadata representing the video path
	
**steps**: Defines the processing steps to be executed. Each step contains a type, name, and specific configuration.

**Adding New Steps**

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

## 2 Input and Output Formats

### 2.1 Input Format

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

### 2.2 Output Format

The output format depends on the steps executed. Typically, results are saved in the folder specified by save_folder in the configuration file, and they are organized by step name.

For caption generation methods, the generated results are saved in the JSONL file under the user-specified key (e.g., ‘text’), while for image generation methods, the images/videos are stored in the specified directory.

### 2.3 Intermediate Results

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
### 2.4 Final Results

The final results are stored in the save_folder. The content depends on the output of the last step. For example, if the last step is image generation, the final results will include the generated image files.

Example directory structure for final results:
```
results/
├── generated_images/
│  ├── image1_generated.jpg
│  ├── image2_generated.png
```

## 3 Examples

### 3.1 Prepare Data

Create a JSONL file with image captions, for example, test_image_captioner.jsonl:

{"image": "images/cat.jpg", "text": "A cat sitting on a windowsill."}

{"image": "images/sunset.png", "text": "A beautiful sunset over the mountains."}

### 3.2 Run the Pipeline

Use the provided sample configuration file config.yaml to run the pipeline:
```bash
python run_pipeline.py --config configs/ImageCaption.yaml
```
### 3.3 View Results

The generated images and captions will be saved in the intermediate_results/ and results/ directories.

