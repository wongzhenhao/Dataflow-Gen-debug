
[English Readme](./README.md)
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

DataFlow-Gen是一个自动合成多模态数据的工具。 我们主要支持多种文生X，X生文的算法。

我们目前支持文本，图像，视频三个模态。

## Table of Contents
- [DataFlow-Eval](#dataflow-eval)
  - [Table of Contents](#table-of-contents)
  - [Module and Modality Support](#module-and-modality-support)
  - [🔥 News](#news)
  - [🛠 Installation](#installation)
  - [🚀 Quick Start](#quick-start)
  - [💪 Jupyter Notebook Demo](#jupyter-notebook-demo)
    - [Text](#text)
    - [Image](#image)
    - [Video](#video)
  - [📌 Data Evaluation Documentation](#data-evaluation-documentation)
    - [Text Documentation](#text-documentation)
    - [Image Documentation](#image-documentation)
    - [Video Documentation](#video-documentation)
  - [🧠 Data Evaluation Algorithms](#data-evaluation-algorithms)
    - [Text Algorithms](#text-algorithms)
    - [Image Algorithms](#image-algorithms)
    - [Video Algorithms](#video-algorithms)
  - [👋 Awesome Data Evaluation](#awesome-data-evaluation)

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
python run_pipeline.py --config configs/TextGeneration.yaml # Text Generation
python run_pipeline.py --config configs/ImageCaption.yaml # Image Captioning
python run_pipeline.py --config configs/ImageGeneration.yaml # Image Generation
python run_pipeline.py --config configs/VideoCaption.yaml # Video Captioning
python run_pipeline.py --config configs/VideoGeneration.yaml # Video Generation
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

## Awesome Data Generation
- [👋 Awesome Data Generation](./Awesome_Data_Generation.md)
