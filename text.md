# Text Processing Pipeline

### 目录

- [Text Processing Pipeline](#text-processing-pipeline)
    - [目录](#目录)
    - [文本方法概览](#文本方法概览)
    - [使用方法](#使用方法)
    - [输入与输出格式](#输入与输出格式)
    - [示例](#示例)
    - [项目结构](#项目结构)

### 文本方法概览

文本生成部分主要涵盖两大类文本生成算法：**API文本生成**与**本地模型文本生成**。以下内容将详细介绍每类方法的具体模型、功能简介。

**API文本生成方法**

<table>
  <thead>
    <tr>
      <th>名称</th>
      <th>模型接口</th>
      <th>简介</th>
      <th>官方仓库或文档</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>APIGenerator</td>
      <td>使用aisuite格式的模型接口"provider:model"，例如"openai:gpt-4o"</td>
      <td>使用aisuite库的统一接口，从包含OpenAI、Claude、Huggingface在内多个平台，使用APIkey获得模型相应</td>
      <td><a href="https://github.com/andrewyng/aisuite">Github Page of aisuite</a></td>
    </tr>
  </tbody>
</table>

**本地模型文本生成方法**

<table>
  <thead>
    <tr>
      <th>名称</th>
      <th>模型接口</th>
      <th>简介</th>
      <th>官方仓库或文档</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>LocalModelGenerator</td>
      <td>Huggingface上的模型路径名，例如Qwen/Qwen2.5-1.5B-Instruct</td>
      <td>生成器首先将指定模型下载到指定路径，然后基于vllm进行推理，获得模型响应。</td>
      <td>暂无</td>
    </tr>
  </tbody>
</table>

### 使用方法

1. **运行pipeline**:

   执行起始脚本 `run_pipeline.py`。此脚本根据配置文件运行整个文本生成pipeline，包括：

   - **预处理**：从用户指定的jsonl文件中，依据configs中指定的text key，读取每个json中相应的key，作为prompt.
   - **模型推理**：根据用户的配置文件调用模型进行文本生成。
   - **后处理**：完成数据格式转化、结果保存等工作，将内部处理格式转换为用户的格式。

   **命令行运行**

   ```bash
   python run_pipeline.py --config path/to/config.yaml
   ```

2. **配置文件**

   配置文件采用 YAML 格式，用于定义pipeline的输入、输出以及各个步骤的配置。

   **示例配置文件**

   ```yaml
   meta_path: data/text/test_text_generation.jsonl # 元数据路径，包含文本生成的提示
   base_folder: text_intermediate_results/ # 保存中间结果的路径
   save_folder: results/ # 保存生成文本的路径
   text_key: prompt # 元数据中文本提示的键

   steps:
     - type: TextGenerator
       name: APIGenerator # API KEY必须在环境变量中设置
       config:
         model_id: "openai:gpt-4o"
         temperature: 0.75
         top_p: 0.95
         max_tokens: 20
         n: 1 # 当前仅支持 n = 1
         stream: False
         stop: null
         presence_penalty: 0.0
         frequency_penalty: 0.0
         prompt: "You are a helpful assistant."

     - type: TextGenerator
       name: LocalModelGenerator
       config:
         device: "cuda"
         model_path: "Qwen/Qwen2.5-1.5B-Instruct"
         n: 1
         best_of: null
         presence_penalty: 0
         frequency_penalty: 0
         repetition_penalty: 1
         temperature: 1
         top_p: 1
         top_k: -1
         min_p: 0
         seed: null
         stop: null
         stop_token_ids: null
         ignore_eos: False
         max_tokens: 32
         min_tokens: 0
         logprobs: null
         prompt_logprobs: null
         detokenize: True
         skip_special_tokens: True
         spaces_between_special_tokens: True
         logits_processors: null
         include_stop_str_in_output: False
         truncate_prompt_tokens: null
         logit_bias: null # Dict[int,float]
         allowed_token_ids: null  # List[int]
         download_dir: "ckpr/models/"
         prompt: "You are a helpful assistant."
   ```

   **配置参数说明**

   - **meta_path**: 指向包含元数据的JSONL文件的路径。每行一个JSON对象，包含文本生成的提示。
   - **base_folder**: 中间结果保存的根目录。
   - **save_folder**: 最终生成结果的保存目录。
   - **text_key**: 元数据中表示文本提示的键。
   - **steps**: 定义需要执行的步骤。每个步骤包含类型、名称和特定的配置。

   **添加新的生成方法步骤**

   要添加新的文本生成步骤，可以在 `steps` 列表中添加相应的配置。例如，添加一个新的API生成步骤：

   ```yaml
   - type: TextGenerator
     name: APIGenerator
     config:
       model_id: "openai:gpt-4o"
       temperature: 0.7
       top_p: 0.9
       max_tokens: 50
       n: 1
       stream: False
       stop: null
       presence_penalty: 0.0
       frequency_penalty: 0.0
       prompt: "Please generate a detailed summary based on the following input."
   ```

### 输入与输出格式

**1. 输入格式**

支持多种输入输出格式，包括csv、tsv、parquet、json、jsonl格式。框架会将不同输入格式转化为统一的字典进行存储，并且中间结果的输出格式为jsonl，以下以jsonl格式为例进行介绍。

对于输入的JSONL文件，每行包含一个JSON对象，描述单个数据项。JSON对象应包含相应的键。

**生成方法对应的 JSONL 文件**

```json
{"prompt": "请介绍一下人工智能的发展历程。"}
{"prompt": "写一首关于春天的诗。"}
{"prompt": "解释一下量子计算的基本原理。"}
```

**关键字段说明**

- **prompt**: 用于生成文本的提示语。

**2. 输出格式**

输出会保存到配置文件中指定的 `save_folder` 中，并按照步骤名称进行组织。


**3. 中间结果**

中间结果保存在 `base_folder` 中，每个步骤会在此目录下创建一个子文件夹来存储其输出。

例如：

```
text_intermediate_results/
├── step_0_preprocess_format/
│  ├── result.jsonl
├── step_1_APIGenerator/
│  ├── result.jsonl
├── step_2_LocalModelGenerator/
│  ├── result.jsonl
```

**4. 最终结果**

最终结果会保存在 `save_folder` 中，具体内容取决于最后一个步骤的输出。例如，若最后一个步骤是本地模型生成，最终结果包括生成的文本文件。

```
results/
├── generated_texts.jsonl
```

### 示例

1. **准备数据**

   创建一个包含文本提示的JSONL文件，例如 `test_text_generation.jsonl`：

   ```json
   {"prompt": "请介绍一下人工智能的发展历程。"}
   {"prompt": "写一首关于春天的诗。"}
   {"prompt": "解释一下量子计算的基本原理。"}
   ```

2. **运行程序**

   使用提供的示例配置文件 `config.yaml` 运行管道：

   ```bash
   python run_pipeline.py --config config.yaml
   ```
   如果使用APIgenerator，请首先将您的APIkey加载到您的环境变量中。

3. **查看结果**

   生成的文本将保存在 `text_intermediate_results/` 和 `results/` 目录中。

### 项目结构

```
TextGen-Project/
├── data/
│   ├── text/
│   │   ├── test_text_generation.jsonl
├── intermediate_results/
├── results/
├── src/
│   ├── utils/
│   │   ├── registry.py
│   │   ├── data_utils.py
│   ├── generators/
│   │   ├── APIGenerator.py
│   │   ├── LocalModelGenerator.py
│   ├── config.py
│   ├── pipeline/
│   │   ├── manager.py
│   │   ├── steps.py
│   │   ├── wrappers.py
│   ├── data/
│   │   ├── DataManager.py
│   │   ├── Dataset.py
├── run_pipeline.py
├── requirements.txt
├── config.yaml
├── README.md
```

- **data/**: 存放输入数据的目录。
  - **text/**: 包含文本生成相关的数据文件。
- **intermediate_results/**: 存放各步骤中间结果的目录。
- **results/**: 存放最终生成结果的目录。
- **src/**: 源代码目录。
  - **utils/**: 工具模块，如注册表和数据处理工具。
  - **generators/**: 文本生成器模块，包括API和本地模型生成器。
  - **config.py**: 配置相关的代码。
  - **pipeline/**: 管道管理和步骤执行相关的代码。
  - **data/**: 数据管理和数据集相关的代码。
- **run_pipeline.py**: 运行管道的主脚本。
- **requirements.txt**: 项目依赖列表。
- **config.yaml**: 管道配置文件。
- **README.md**: 项目说明文档。

