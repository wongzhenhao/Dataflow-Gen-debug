**Supplemnt**

### 如何将自己的方法添加进仓库?

我们的项目实现了高度的拓展性，使用者可以方便地在 src/model 目录下相应的模态中添加自己的方法，并通过配置 YAML 文件中的模型参数即可运行。以下将详细说明添加方法的步骤。

**方法实现的拓展性**

使用者只需在 src/model 目录下对应的模态文件夹中实现新的方法类，并确保该类包含必要的初始化和主要接口函数。然后，在 YAML 文件中配置相应的模型参数，即可在管道中运行该方法。

**示例：LLaVACaptioner**

以下是 LLaVACaptioner 方法的实现示例：

```
class LLaVACaptioner:
    def __init__(self, 
                 model_path: str = "llava-hf/llava-1.5-7b-hf", 
                 trust_remote_code: bool = True,
                 tensor_parallel_size: int = 1,
                 max_model_len: int = 256,
                 max_num_seqs: int = 128, 
                 temperature: float = 0.6,
                 top_p: float = 0.9,
                 top_k: int = 50,
                 repetition_penalty: float = 1.2,
                 prompt: str = "What is the content of this image?",
                 **kwargs):
        logging.info(f"VLLM model LLaVACaptioner will initialize with model_path: {model_path}")
        self.model = LLM(model=model_path, trust_remote_code=trust_remote_code, tensor_parallel_size=tensor_parallel_size, max_model_len=max_model_len, max_num_seqs=max_num_seqs)
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_model_len,
        )
        self.prompt = "USER: <image>\n" + prompt + "\nASSISTANT:"

    def generate_batch(self, images):
        inputs, outputs = [], []
        for image in images:
            inputs.append({
                "prompt": self.prompt,
                "multi_modal_data": {
                    "image": self.encode_images(image),
                },
            })
        response = self.model.generate(inputs, self.sampling_params)
        for r in response:
            outputs.append(r.outputs[0].text.strip())

        return outputs

    def encode_images(self, image):
        # 实现图像编码的方法
        pass
```

**初始化模型**

在 __init__ 方法中，完成模型的初始化以及运行所需的参数配置。

**主要接口函数**

generate_batch 方法是供管道框架调用的主要接口，输入可以是图片地址、视频地址、文本、描述等。需确保输入内容与模型功能对应。此外，可以设计辅助方法如 encode_images 来支持 generate_batch 函数的实现。

**配置 YAML 参数文件**

编写完模型后，仅需根据 __init__ 方法中的参数编写相应的 YAML 参数文件。配置文件应包含输入输出路径、键值对、需要运行的模型步骤（step）等内容。

示例 YAML 配置：

```
model:
  type: LLaVACaptioner
  model_path: "llava-hf/llava-1.5-7b-hf"
  trust_remote_code: true
  tensor_parallel_size: 1
  max_model_len: 256
  max_num_seqs: 128
  temperature: 0.6
  top_p: 0.9
  top_k: 50
  repetition_penalty: 1.2
  prompt: "What is the content of this image?"

steps:
  - name: LLaVACaptioner
    inputs:
      - image_path
    outputs:
      - caption
```



我们鼓励实现更多的方法并通过 Pull Request（PR）向我们反馈!

### YAML 文件中的多个 Step 如何调用?

在一个 YAML 文件中，可以定义多个步骤（step），这些步骤按照严格的顺序调用，并具有因果关系。下一个步骤的输入是上一个步骤的输出。

**步骤顺序与因果关系**

确保每个步骤按照逻辑顺序执行。例如，先生成描述，再根据描述生成图片：

​	1.	**ImageCaptioner**：对图像进行描述生成。

​	2.	**ImageGenerator**：根据描述生成新的图像。

**示例：ImageCaptioner -> ImageGenerator**

```
steps:
  - name: ImageCaptioner
    inputs:
      - image_path
    outputs:
      - caption

  - name: ImageGenerator
    inputs:
      - caption
    outputs:
      - generated_image
```

**多数据集处理**

一个 YAML 文件主要用于对一个数据集按照逻辑顺序或生成顺序进行处理。如果需要对多个数据集进行处理，请编写多个 YAML 文件，每个文件对应一个数据集。

**示例 YAML 文件**

在 examples 目录中，我们提供了结合 SOTA 方法及不同论文的示例 YAML 文件，如**ALLaVA**、**Q-Instruct**、 **SynthVLM**等方法。

