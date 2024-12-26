# Image Processing Pipeline

### 目录

- [图片方法概览](#图片方法概览)
  - [图片描述生成方法](#图片描述生成方法)
  - [图片生成方法](#图片生成方法)
- [使用方法](#使用方法)
  - [运行程序](#运行程序)
  - [配置文件](#配置文件)
  - [添加新的方法步骤](#添加新的方法步骤)
- [输入与输出格式](#输入与输出格式)
- [示例](#示例)
  - [准备数据](#准备数据)
  - [运行程序](#运行程序-1)
  - [查看结果](#查看结果)

### 图片方法概览

图片部分主要涵盖两大类图片处理算法：**图片描述生成**与**图片生成**。以下内容将详细介绍每类方法的具体模型、功能简介。

**图片描述生成方法**

<table>
  <thead>
    <tr>
      <th>名称</th>
      <th>模型huggingface路径</th>
      <th>简介</th>
      <th>官方仓库或论文</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>BLIPCaptioner</td>
      <td>Salesforce/blip2-opt-2.7b</td>
      <td>使用基于QFormer的 BLIP2 模型生成图像描述，该方法使用广泛，例如LLaVA Pretrain 558
        数据集。</td>
      <td><a href="https://github.com/salesforce/LAVIS/tree/main/projects/blip2">代码</a><br><a href="https://arxiv.org/abs/2301.12597">论文</a></td>
    </tr>
    <tr>
      <td>LLaVACaptioner</td>
      <td>llava-hf/llava-1.5-7b-hf</td>
      <td>基于 LLaVA 模型生成图像描述，适用于简单任务情况。</td>
      <td><a href="https://github.com/haotian-liu/LLaVA">代码</a><br><a href="https://arxiv.org/abs/2304.08414">论文</a></td>
    </tr>
    <tr>
      <td>LLaVANeXTCaptioner</td>
      <td>llava-hf/llava-v1.6-mistral-7b-hf</td>
      <td> 基于动态分辨率的 LLaVA-NeXT 模型，提升了语义复杂度的图像描述生成能力。</td>
      <td><a href="https://github.com/LLaVA-VL/LLaVA-NeXT">代码</a><br><a href="https://llava-vl.github.io/blog/2024-01-30-llava-next/">博客</a></td>
    </tr>
    <tr>
      <td>LLaVAOneVisionCaptioner</td>
      <td>llava-hf/llava-onevision-qwen2-7b-ov-hf</td>
      <td>基于 OneVision 版本的 LLaVA 模型，在大量数据集上进行训练，优化了视觉信息处理。</td>
      <td><a href="https://github.com/LLaVA-VL/LLaVA-NeXT">代码</a><br><a href="https://arxiv.org/abs/2408.03326">论文</a></td>
    </tr>
    <tr>
      <td>MLLamaCaptioner</td>
      <td>meta-llama/Llama-3.2-11B-Vision-Instruct</td>
      <td>基于 Meta 的 LLaMA Vision模型，支持高并行度的图像描述生成。</td>
      <td><a href="https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/">官网</a></td>
    </tr>
    <tr>
      <td>Phi3VCaptioner</td>
      <td>microsoft/Phi-3-vision-128k-instruct</td>
      <td>基于微软的 Phi-3 模型，专注于高质量的指令式图像描述生成。</td>
      <td><a href="https://github.com/microsoft/Phi-3CookBook">代码</a><br><a href="https://azure.microsoft.com/en-us/blog/introducing-phi-3-redefining-whats-possible-with-slms/">官网</a><br><a href="https://arxiv.org/abs/2404.14219">论文</a></td>
    </tr>
    <tr>
      <td>QwenVLCaptioner</td>
      <td>Qwen/Qwen2-VL-7B-Instruct</td>
      <td>基于 Qwen2 模型，支持多轮对话式的图像描述生成。</td>
      <td><a href="https://github.com/QwenLM/Qwen2.5">代码</a><br><a href="https://arxiv.org/abs/2407.10671">论文</a></td>
    </tr>
  </tbody>
</table>

**图片生成方法**

<table>
  <thead>
    <tr>
      <th>名称</th>
      <th>模型huggingface路径</th>
      <th>简介</th>
      <th>官方仓库或论文</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>FLUXGenerator</td>
      <td>black-forest-labs/FLUX.1-dev</td>
      <td>基于 Flux 实现的图片生成方法，支持高分辨率图像生成，指令跟随能力较强。</td>
      <td><a href="https://github.com/black-forest-labs/flux">代码</a></td>
    </tr>
    <tr>
      <td>StableDiffusionGenerator</td>
      <td>stabilityai/stable-diffusion-3.5-large</td>
      <td>基于 Stable Diffusion 3 实现的图片生成方法，能够生成高质量、细节丰富的图像，速度较快。</td>
      <td><a href="https://github.com/Stability-AI/StableDiffusion">代码</a><br><a href="https://arxiv.org/abs/2112.10752">论文</a></td>
    </tr>
  </tbody>
</table>

### 使用方法

1. **运行pipeline**: 

执行起始脚本是 run_pipeline.py。此脚本根据配置文件运行整个图像处理pipeline，包括：

-  预处理：完成数据格式转换、数据初始化等工作，把用户的格式转化成内部处理格式。
- 模型推理：根据用户的yaml文件依次调用不同模型进行生成。
- 后处理：完成数据格式转化、结果保存等工作，把内部处理格式转换成用户的格式。

**命令行运行**

```bash
python run_pipeline.py --config path/to/config.yaml
```

2. **配置文件**

配置文件采用 YAML 格式，用于定义pipeline的输入、输出以及各个步骤的配置。

**示例配置文件**

路径: examples/image_demo.yaml

```yaml
meta_path: data/image/test_image_captioner.jsonl # 元数据路径，图像和视频的固定格式
meta_folder: data/image # 仅适用于描述生成器
base_folder: intermediate_results/ # 保存中间结果的路径
save_folder: results/ # 保存生成图像的路径
image_key: image # 元数据中图像的键
text_key: text # 元数据中文本的键
video_key: video # 元数据中视频的键

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
 -type: other
```

**配置参数说明**

​	•	**meta_path**: 指向包含元数据的JSONL文件的路径。每行一个JSON对象，包含图像或视频的相关信息。

​	•	**meta_folder**: 图像或视频文件所在的文件夹路径。

​	•	**base_folder**: 中间结果保存的根目录。

​	•	**save_folder**: 最终生成结果的保存目录。

​	•	**image_key**: 元数据中表示图像路径的键。

​	•	**text_key**: 元数据中表示文本内容的键。

​	•	**video_key**: 元数据中表示视频路径的键。

​	•	**steps**: 定义需要执行的步骤。每个步骤包含类型、名称和特定的配置。

**添加新的生成步骤**

要添加新的处理步骤，可以在steps列表中添加相应的配置。例如，添加一个新的文本生成步骤：

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

### 输入与输出格式

**1. 输入格式**

支持多种输入输出格式，包括csv、tsv、parquet、json、jsonl格式，框架会把不同输入格式转化成统一的字典进行存储，并且中间结果的输出格式为jsonl，下面以jsonl格式为例进行介绍。

对于输入的JSONL文件，每行包含一个JSON对象，描述单个数据项。根据不同的处理类型，JSON对象应包含相应的键。

**生成方法对应的 JSONL 文件**

{"image": "image1.jpg", "text": "A cat sitting on a windowsill."}

{"image": "image2.png", "text": "A beautiful sunset over the mountains."}

{"video": "video1.mp4", "text": "A timelapse of city traffic at night."}

**关键字段说明**

​	•	**image**: 图像文件的的名称。

​	•	**video**: 视频文件的名称。

​	•	**text**: 与图像或视频相关的文本描述或提示。

描述生成方法的格式与生成方法的格式类似，只需指明需要生成的描述的图片\视频在相应目录下的名称。

**2. 输出格式**

输出结果根据执行的步骤不同而有所不同。通常，输出会保存到配置文件中指定的save_folder中，并按照步骤名称进行组织。

对于描述生成方法，生成结果保存在jsonl文件用户指定的键中，例如 'text'，对于生成方法，图片\视频存放在用户指定的目录下，图片\视频的名称由用户指定。

**3. 中间结果**

中间结果保存在base_folder中，每个步骤会在此目录下创建一个子文件夹来存储其输出。

例如：

intermediate_results/

├── step_0_preprocess_format/

│  ├── result.jsonl

├── step_1_ImageCaptioner_LLaVACaptioner/

│  ├── result.jsonl

├── step_2_ImageGenerator_FLUXGenerator/

│  ├── generated_images/

│    ├── image1_generated.jpg

│    ├── image2_generated.png

**4. 最终结果**

最终结果会保存在save_folder中，具体内容取决于最后一个步骤的输出。例如，若最后一个步骤是图像生成，最终结果包括生成的图像文件。

results/

├── generated_images/

│  ├── image1_generated.jpg

│  ├── image2_generated.png

### 示例

1. **准备数据**

创建一个包含图像描述的JSONL文件，例如test_image_captioner.jsonl：

{"image": "images/cat.jpg", "text": "A cat sitting on a windowsill."}

{"image": "images/sunset.png", "text": "A beautiful sunset over the mountains."}

2. **运行程序**

使用提供的示例配置文件config.yaml运行程序：

python run_pipeline.py --config examples/image_demo.yaml

3. **查看结果**

生成的图像和描述将保存在intermediate_results/和results/目录中。

### 项目结构

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

