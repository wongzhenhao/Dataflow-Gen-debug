[中文主页](./README.zh-CN.md)
# Dataflow-Gen

<p align="center">
  <img src="./static/images/DataFlow_gen.png">
</p>
<a href="https://opensource.org/license/apache-2-0" target="_blank">
    <img alt="License: apache-2-0" src="https://img.shields.io/github/license/saltstack/salt" />
</a>
<a href="https://github.com/Open-DataFlow/Dataflow-Gen" target="_blank">
    <img alt="GitHub Stars" src="https://img.shields.io/github/stars/Open-DataFlow/Dataflow-Gen?style=social" />
</a>
<a href="https://github.com/Open-DataFlow/Dataflow-Gen/issues" target="_blank">
    <img alt="Open Issues" src="https://img.shields.io/github/issues-raw/Open-DataFlow/Dataflow-Gen" />
</a>

DataFlow-Gen is a data generation system to generate high-quality data automatically. We mainly support SOTA algorithms within academic papers with strong theoretical support.

We now support text, image, video, and multimodality data types.

## Table of Contents
- [DataFlow-Gen](#dataflow-gen)
  - [Table of Contents](#table-of-contents)
  - [Module and Modality Support](#module-and-modality-support)
  - [🔥 News](#news)
  - [🛠 Installation](#installation)
  - [🚀 Quick Start](#quick-start)
  - [📌 Data Generation Documentation](#data-generation-documentation)
    - [Text Documentation](#text-documentation)
    - [Image Documentation](#image-documentation)
    - [Video Documentation](#video-documentation)

## Module and Modality Support

| Module\Modality     | Text | Image | Video | Image-Text Pair | Video-Text Pair |
| ------------------- | ---- | ----- | ----- | --------------- | --------------- |
| **Data Evaluation** | ✅    | ✅     | ✅     | ✅               | ✅               |

## News
- [2024-12-27] 🎉 Our first data generation system is now open source.

## Installation
```bash
conda create -n dataflow-gen python=3.10 -y
conda activate dataflow-gen
pip install -r requirements.txt
```
## Quick Start
```
cd path/to/DataFlow-Gen
python run_pipeline.py --config configs/TextGeneration.yaml   # Text Generation
python run_pipeline.py --config configs/ImageCaption.yaml     # Image Captioning
python run_pipeline.py --config configs/ImageGeneration.yaml  # Image Generation
python run_pipeline.py --config configs/VideoCaption.yaml     # Video Captioning
python run_pipeline.py --config configs/VideoGeneration.yaml  # Video Generation
```


## Data Generation Documentation

For the usage of evaluation, please refer to the following documents👇

### Text Documentation

- [Text Data Generation User Documentation (English)](./Dataflow-Gen/docs/text.md)
- [文本数据生成使用文档 (中文)](./Dataflow-Gen/docs/text.zh-CN.md)

### Image Documentation

- [Image Data Generation User Documentation (English)](./Dataflow-Gen/docs/image.md)
- [图像数据生成使用文档 (中文)](./Dataflow-Gen/docs/image.zh-CN.md)

### Video Documentation

- [Video Data Generation User Documentation (English)](./Dataflow-Gen/docs/video.md)
- [视频数据生成使用文档 (中文)](./Dataflow-Gen/docs/video.zh-CN.md)

