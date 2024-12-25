# Video Processing Pipeline

### 目录

- [视频方法概览](#视频方法概览)
  - [视频描述生成方法](#视频描述生成方法)
  - [视频生成方法](#视频生成方法)
- [使用方法](#使用方法)
  - [运行程序](#运行程序)
  - [配置文件](#配置文件)
  - [添加新的方法步骤](#添加新的方法步骤)
- [输入与输出格式](#输入与输出格式)
- [示例](#示例)
  - [准备数据](#准备数据)
  - [运行程序](#运行程序-1)
  - [查看结果](#查看结果)

### 视频方法概览

视频部分主要涵盖两大类视频处理算法：**视频描述生成**与**视频生成**。以下内容将详细介绍每类方法的具体模型、功能简介。

**视频描述生成方法**

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
      <td>QwenVLCaptioner</td>
      <td>Qwen/Qwen2-VL-7B-Instruct</td>
      <td>基于 Qwen2-VL 模型生成视频描述，有不同大小的模型，适用不同粒度的视频描述。</td>
      <td><a href="https://github.com/QwenLM/Qwen2-VL">代码</a><br><a href="https://arxiv.org/pdf/2409.12191">论文</a></td>
    </tr>
    <tr>
      <td>LLaVANeXTVideoCaptioner</td>
      <td>llava-hf/LLaVA-NeXT-Video-7B-hf</td>
      <td> 基于动态分辨率的 LLaVA-NeXT 模型，提升了语义复杂度的视频描述生成能力。</td>
      <td><a href="https://github.com/LLaVA-VL/LLaVA-NeXT">代码</a><br><a href="https://llava-vl.github.io/blog/2024-01-30-llava-next/">博客</a></td>
    </tr>
    <tr>
      <td>LLaVAOVCaptioner</td>
      <td>llava-hf/llava-onevision-qwen2-7b-ov-hf</td>
      <td>基于 OneVision 版本的 LLaVA 模型，在大量数据集上进行训练，优化了视觉信息处理。</td>
      <td><a href="https://github.com/LLaVA-VL/LLaVA-NeXT">代码</a><br><a href="https://arxiv.org/abs/2408.03326">论文</a></td>
    </tr>
  </tbody>
</table>

**视频生成方法**

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
      <td>基于 Flux 实现的视频生成方法，支持高分辨率图像生成，指令跟随能力较强。</td>
      <td><a href="https://github.com/black-forest-labs/flux">代码</a></td>
    </tr>
    <tr>
      <td>StableDiffusionGenerator</td>
      <td>stabilityai/stable-diffusion-3.5-large</td>
      <td>基于 Stable Diffusion 3 实现的视频生成方法，能够生成高质量、细节丰富的图像，速度较快。</td>
      <td><a href="https://github.com/Stability-AI/StableDiffusion">代码</a><br><a href="https://arxiv.org/abs/2112.10752">论文</a></td>
    </tr>
    <tr>
      <td>AllegroGenerator</td>
      <td>rhymes-ai/Allegro</td>
      <td>能够根据简单的文本输入生成高质量、动态的视频，时长可达 6 秒，帧率为 15FPS，分辨率为 720p。</td>
      <td><a href="https://github.com/rhymes-ai/Allegro">代码</a><br><a href="https://arxiv.org/abs/2410.15458">论文</a></td>
    </tr>
    <tr>
      <td>AnimateDiffGenerator</td>
      <td>guoyww/animatediff-motion-adapter-v1-5-2</td>
      <td>基于Stable Diffusion及 DreamBooth、LoRA 等个性化技术实现的动画生成框架，能够为多数现有的个性化文本到图像模型一键植入动画能力</td>
      <td><a href="https://github.com/guoyww/AnimateDiff">代码</a><br><a href="https://arxiv.org/abs/2307.04725">论文</a></td>
    </tr>
    <tr>
      <td>CogVideoXT2VGenerator</td>
      <td>rhymes-ai/Allegro</td>
      <td>基于diffusion transformer构建的文本到视频生成模型，在运动表现、时长和文本-视频对齐方面优势显著</td>
      <td><a href="https://github.com/THUDM/CogVideo">代码</a><br><a href="https://arxiv.org/abs/2408.06072">论文</a></td>
    </tr>
    <tr>
      <td>CogVideoXGenerator</td>
      <td>THUDM/CogVideoX-5b-I2V</td>
      <td>基于diffusion transformer构建的图像到视频生成模型，在运动表现、时长和文本-视频对齐方面优势显著</td>
      <td><a href="https://github.com/THUDM/CogVideo">代码</a><br><a href="https://arxiv.org/abs/2408.06072">论文</a></td>
    </tr>
    <tr>
      <td>I2VGenXLGenerator</td>
      <td>ali-vilab/i2vgen-xl</td>
      <td>基于层级编码与扩散模型优化实现的视频生成方式，，最终能够依据用户输入从单张静态图像生成高质量、逼真动画且时间连贯的高清视频。</td>
      <td><a href="https://github.com/ali-vilab/VGen">代码</a><br><a href="https://arxiv.org/abs/2311.04145">论文</a></td>
    </tr>
    <tr>
      <td>ModelScopeT2VGenerator</td>
      <td>damo-vilab/text-to-video-ms-1.7b</td>
      <td>基于 Stable Diffusion 演进而来的文本到视频合成模型，通过集成时空模块确保帧生成与运动转换效果，具备适应不同帧数的能力从而适用于多种数据集。</td>
      <td><a href="https://github.com/ali-vilab/VGen">代码</a><br><a href="https://arxiv.org/abs/2308.06571">论文</a></td>
    </tr>
    <tr>
      <td>SVDGenerator</td>
      <td>stabilityai/stable-video-diffusion-img2vid-xt</td>
      <td>基于diffusion构建的视频生成方式，其模型在文本到视频、图像到视频生成及多视图任务中表现优异，可生成高质量且运动表现出色的视频。</td>
      <td><a href="https://github.com/Stability-AI/generative-models">代码</a><br><a href="https://arxiv.org/abs/2311.15127">论文</a></td>
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
python run_pipeline.py --config ./configs/VideoCaption.yaml
```

2. **配置文件**

配置文件采用 YAML 格式，用于定义pipeline的输入、输出以及各个步骤的配置。

**示例配置文件**

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

{"id": "000000000", "video": "demo/1.mp4"}
{"id": "000000001", "video": "demo/2.mp4"}
{"id": "000000002", "video": "demo/3.mp4"}

**关键字段说明**

​	•	**image**: 图像文件的的名称。

​	•	**video**: 视频文件的名称。

​	•	**text**: 与图像或视频相关的文本描述或提示。

描述生成方法的格式与生成方法的格式类似，只需指明需要生成的描述的视频\视频在相应目录下的名称。

**2. 输出格式**

输出结果根据执行的步骤不同而有所不同。通常，输出会保存到配置文件中指定的save_folder中，并按照步骤名称进行组织。

对于描述生成方法，生成结果保存在jsonl文件用户指定的键中，例如 'text'，对于生成方法，视频\视频存放在用户指定的目录下，视频\视频的名称由用户指定。

**3. 中间结果**

中间结果保存在base_folder中，每个步骤会在此目录下创建一个子文件夹来存储其输出。

例如：

intermediate_results/

├── step_0_preprocess_format/

│  ├── result.jsonl

├── step_1_VideoCaptioner_Qwen2VLCaptioner/

│  ├── result.jsonl

├── step_2_VideoGenerator_FLUXGenerator/

│  ├── generated_images/

│    ├── image1_generated.jpg

│    ├── image2_generated.png

**4. 最终结果**

最终结果会保存在save_folder中，具体内容取决于最后一个步骤的输出。例如，若最后一个步骤是视频生成，最终结果包括生成的视频文件。

results/

├── generated_images/

│  ├── image1_generated.jpg

│  ├── image2_generated.png

### 示例

1. **准备数据**

创建一个包含视频描述的JSONL文件，例如test_video_captioner.jsonl：

{"id": "000000000", "video": "demo/1.mp4"}
{"id": "000000001", "video": "demo/2.mp4"}
{"id": "000000002", "video": "demo/3.mp4"}

2. **运行程序**

使用提供的示例配置文件config.yaml运行管道：

python run_pipeline.py --config ./configs/VideoCaption.yaml

3. **查看结果**

生成的视频和描述将保存在intermediate_results/和results/目录中。

### 项目结构

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
