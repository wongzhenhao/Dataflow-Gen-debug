## Video Data Evaluation

### 1. Pure Video Data Evaluation

#### 1.1 Dataset Preparation
Users can store the metadata of their dataset in the following JSON format:
```json
[
{"id": "000000000", "video": "/mnt/hwfile/mllm/niujunbo/VideoBench/AutoEvalMetaData/0.mp4"}
{"id": "000000001", "video": "/mnt/hwfile/mllm/niujunbo/VideoBench/AutoEvalMetaData/1.mp4"}
]
```

#### 1.2 Writing the YAML Configuration File

For the dataset from section 1.1, write a YAML file in the following format. The `data` section specifies the dataset path and related information
```yaml
meta_path: './video.jsonl' # path for the meta data, the format is fixed for image and video
meta_folder: data/video # only for captioner
base_folder: intermediate_results/ # path to save the intermediate results
save_folder: results/ # path to save the caption
image_key: image # key for the image in the meta data
text_key: text # key for the text in the meta data
video_key: video # key for the video in the meta data

steps:
  - type: VideoCaptioner
    name: Qwen2VLCaptioner
    config:
      model_path: '../ckpt' # Path to cache models
      trust_remote_code: true
      tensor_parallel_size: 1
      max_model_len: 2048
      max_num_seqs: 128
      temperature: 0.7
      top_p: 0.9
      top_k: 50
      repetition_penalty: 1.2
      prompt: "Please describe the video in detail."
      repeat_time: 5
      batch_size: 1000
```

#### 1.2 Generate caption for the video
Once the YAML configuration file is ready, call the function `calculate_score()` to evaluate the data.

```python
from dataflow.utils.utils import calculate_score
calculate_score()
```
output:
```
{
    'meta_scores': {}, 
    'item_scores': 
    {
        '0': 
        {
            'VideoMotionScorer': {'Default': 0.6842129230499268}
        }, 
        '1': 
        {
            'VideoMotionScorer': {'Default': 8.972004890441895}
        }
    }
}
```


### 2. Video-Text Data Evaluation

#### 2.1 Dataset Preparation

Users can store the metadata of their dataset in the following JSON format:

```json
[
    {
        "video": "test_video.avi",
        "captions": [
            "A man is clipping paper.", 
            "A man is cutting paper."
        ]
    }
]
```

#### 2.2 Writing the YAML Configuration File

For the dataset from section 2.1, write a YAML file in the following format. The `data` section specifies the dataset path and related information, while the `scorers` section defines the evaluation metrics to be used.

```yaml
model_cache_path: '../ckpt' # Path to cache models
num_workers: 2

data:
  video:
    meta_data_path: './video-caption.json' # Path to meta data (mainly for image or video data)
    data_path: './' # Path to dataset
    formatter: 'VideoCaptionFormatter' # Formatter for video-text evaluation

scorers:
  EMScorer:
    batch_size: 4
    num_workers: 4
```

#### 2.3 Evaluating the Dataset
Once the YAML configuration file is ready, call the function `calculate_score()` to evaluate the data.

```python
from dataflow.utils.utils import calculate_score
calculate_score()
```