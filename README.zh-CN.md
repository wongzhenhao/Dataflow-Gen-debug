
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

## 目录
- [DataFlow-Gen](#dataflow-gen)
  - [目录](#目录)
  - [支持的模态](#支持的模态)
  - [🔥 新闻](#新闻)
  - [🛠 安装](#安装)
  - [🚀 快速开始](#快速开始)
  - [📌 数据生成文档](#数据生成文档)
    - [Text Documentation](#text-documentation)
    - [Image Documentation](#image-documentation)
    - [Video Documentation](#video-documentation)

## 支持的模态

| Module\Modality     | Text | Image | Video | Image-Caption | Video-Caption |
| ------------------- | ---- | ----- | ----- | --------------- | --------------- |
| **Data Evaluation** | ✅    | ✅     | ✅     | ✅               | ✅               |

## 新闻
- [2024-12-27] 🎉 我们开源了数据生成系统

## 安装
```bash
conda create -n dataflow-gen python=3.10 -y
conda activate dataflow-gen
pip install -r requirements.txt
```
## 快速开始
```
cd path/to/DataFlow-Gen
python run_pipeline.py --config configs/TextGeneration.yaml # Text Generation
python run_pipeline.py --config configs/ImageCaption.yaml # Image Captioning
python run_pipeline.py --config configs/ImageGeneration.yaml # Image Generation
python run_pipeline.py --config configs/VideoCaption.yaml # Video Captioning
python run_pipeline.py --config configs/VideoGeneration.yaml # Video Generation
```

## 数据生成文档

For the usage of evaluation, please refer to the following documents👇

### 文本模态

- [Text Data Generation User Documentation (English)](./Dataflow-Gen/docs/text.md)
- [文本数据生成使用文档 (中文)](./Dataflow-Gen/docs/text.zh-CN.md)

### 图像模态

- [Image Data Generation User Documentation (English)](./Dataflow-Gen/docs/image.md)
- [图像数据生成使用文档 (中文)](./Dataflow-Gen/docs/image.zh-CN.md)

### 视频模态

- [Video Data Generation User Documentation (English)](./Dataflow-Gen/docs/video.md)
- [视频数据生成使用文档 (中文)](./Dataflow-Gen/docs/video.zh-CN.md)

