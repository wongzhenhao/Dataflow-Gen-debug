# Text Processing Pipeline

### Table of Contents

- [Overview of Text Methods](#overview-of-text-methods)
- [Usage Instructions](#usage-instructions)
- [Input and Output Formats](#input-and-output-formats)
- [Examples](#examples)
- [Project Structure](#project-structure)

### Overview of Text Methods

The text generation section mainly covers two categories of text generation algorithms: **API-based Text Generation** and **Local Model Text Generation**. The following details the specific models and functionalities of each method.

**API-based Text Generation Methods**

<table>
  <thead>
    <tr>
      <th>Name</th>
      <th>Model Interface</th>
      <th>Description</th>
      <th>Official Repository or Documentation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>APIGenerator</td>
      <td>Uses aisuite-style model interface "provider:model", e.g., "openai:gpt-4o"</td>
      <td>Leverages the unified interface of the aisuite library to access responses from multiple platforms including OpenAI, Claude, and Huggingface via API keys.</td>
      <td><a href="https://github.com/andrewyng/aisuite">Github Page of aisuite</a></td>
    </tr>
  </tbody>
</table>

**Local Model Text Generation Methods**

<table>
  <thead>
    <tr>
      <th>Name</th>
      <th>Model Interface</th>
      <th>Description</th>
      <th>Official Repository or Documentation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>LocalModelGenerator</td>
      <td>Model path name on Huggingface, e.g., Qwen/Qwen2.5-1.5B-Instruct</td>
      <td>The generator first downloads the specified model to the designated path and then performs inference based on vllm to obtain model responses.</td>
      <td>Not available</td>
    </tr>
  </tbody>
</table>

### Usage Instructions

1. **Run the Pipeline**:

   Execute the entry script `run_pipeline.py`. This script runs the entire text generation pipeline based on the configuration file, including:

   - **Preprocessing**: Reads the specified text key from each JSON in the user-specified JSONL file as the prompt, based on the configuration in `configs`.
   - **Model Inference**: Generates text using models specified in the configuration file.
   - **Postprocessing**: Converts data formats, saves results, and transforms the internal processing format to the user’s desired format.

   **Command Line Execution**

   ```bash
    python run_pipeline.py --config configs/TextGeneration.yaml
   ```

2. **Configuration File**

   The configuration file uses YAML format to define the pipeline's input, output, and the configuration for each step.

   **Example Configuration File**

   ```yaml
   meta_path: data/text/test_text_generation.jsonl # Path to metadata containing prompts for text generation
   base_folder: text_intermediate_results/ # Path to save intermediate results
   save_folder: results/ # Path to save generated texts
   text_key: prompt # Key in metadata for text prompts

   steps:
     - type: TextGenerator
       name: APIGenerator # API KEY must be set in environment variables
       config:
         model_id: "openai:gpt-4o"
         temperature: 0.75
         top_p: 0.95
         max_tokens: 20
         n: 1 # Currently supports only n = 1
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
   
   **Configuration Parameter Descriptions**

   - **meta_path**: Path to the JSONL file containing metadata. Each line is a JSON object with prompts for text generation.
   - **base_folder**: Root directory to save intermediate results.
   - **save_folder**: Directory to save the final generated results.
   - **text_key**: Key in the metadata that specifies the text prompt.
   - **steps**: Defines the steps to execute. Each step includes the type, name, and specific configurations.
   - **prompt**: The system prompt. In actual usage, the system prompt will be concatenated with the prompt from the JSONL file to form the input to the model.

   **Adding New Text Generation Steps**

   To add a new text generation step, include the corresponding configuration in the `steps` list. For example, adding a new API generation step:

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

### Input and Output Formats

**1. Input Format**

Supports multiple input and output formats, including `csv`, `tsv`, `parquet`, `json`, and `jsonl` formats. The framework converts different input formats into a unified dictionary format for storage, and intermediate results are output in `jsonl` format. The following uses the `jsonl` format as an example.

For input JSONL files, each line contains a JSON object that describes a single data item. The JSON object should include the corresponding keys.

**JSONL File for Generation Methods**

```json
{"prompt": "Please introduce the development history of artificial intelligence."}
{"prompt": "Write a poem about spring."}
{"prompt": "Explain the basic principles of quantum computing."}
```

**Key Field Descriptions**

- **prompt**: The prompt used for text generation.

**2. Output Format**

The output will be saved in the `save_folder` specified in the configuration file and organized by step names.

**3. Intermediate Results**

Intermediate results are saved in the `base_folder`, with each step creating a subfolder under this directory to store its output.

For example:

```
text_intermediate_results/
├── step_0_preprocess_format/
│  ├── result.jsonl
├── step_1_APIGenerator/
│  ├── result.jsonl
├── step_2_LocalModelGenerator/
│  ├── result.jsonl
```

**4. Final Results**

The final results will be saved in the `save_folder`, and the content will depend on the output of the last step. For example, if the last step is local model generation, the final results will include the generated text file.

```
results/
├── generated_texts.jsonl
```

### Examples

1. **Prepare Data**

   Create a JSONL file containing text prompts, for example, `test_text_generation.jsonl`:

   ```json
   {"prompt": "Please introduce the development history of artificial intelligence."}
   {"prompt": "Write a poem about spring."}
   {"prompt": "Explain the basic principles of quantum computing."}
   ```

2. **Run the Program**

   Run the pipeline using the provided example configuration file `config.yaml`:

   ```bash
   python run_pipeline.py --config examples/TextGenerator.yaml
   ```
   
   If using `APIGenerator`, make sure to load your API key into your environment variables first.

3. **Run Process**

   The program first reads the JSONL file specified in the YAML configuration, then initializes the specified Generators. Each Generator combines the system prompt from the YAML configuration with the prompt from the JSONL file, sends it to the respective model to obtain a response, and then saves the response to the designated folder as specified in the YAML configuration.

4. **View Results**

   The generated texts will be saved in the `text_intermediate_results/` and `results/` directories.

### Project Structure

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

- **data/**: Directory for storing input data.
  - **text/**: Contains data files related to text generation.
- **intermediate_results/**: Directory for storing intermediate results of each step.
- **results/**: Directory for storing final generated results.
- **src/**: Source code directory.
  - **utils/**: Utility modules, such as registry and data processing tools.
  - **generators/**: Text generator modules, including API-based and local model generators.
  - **config.py**: Code related to configuration handling.
  - **pipeline/**: Code for pipeline management and step execution.
  - **data/**: Code for data management and dataset handling.
- **run_pipeline.py**: Main script to run the pipeline.
- **requirements.txt**: List of project dependencies.
- **config.yaml**: Pipeline configuration file.
- **README.md**: Project documentation.

