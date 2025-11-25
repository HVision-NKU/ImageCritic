# ImageCritic

> **The Consistency Critic: Correcting Inconsistencies in Generated Images via
Reference-Guided Attentive Alignment**

<a href="https://ouyangziheng.github.io/ImageCritic-Page/"><img src="https://img.shields.io/badge/arXiv-ImageCritic-red" alt="HuggingFace"></a>
<a href="https://ouyangziheng.github.io/ImageCritic-Page/"><img src="https://img.shields.io/badge/Project%20Page-ImageCritic-blue" alt="HuggingFace"></a>
<a href="https://huggingface.co/spaces/ziheng1234/ImageCritic"><img src="https://img.shields.io/badge/🤗_HuggingFace-Space-ffbd45.svg" alt="HuggingFace"></a>
<a href="https://huggingface.co/ziheng1234/ImageCritic"><img src="https://img.shields.io/badge/🤗_HuggingFace-Model-ffbd45.svg" alt="HuggingFace"></a>
<a href="https://huggingface.co/datasets/ziheng1234/Critic-10K"><img src="https://img.shields.io/badge/🤗_HuggingFace-Dataset-ffbd45.svg" alt="HuggingFace"></a>

<img src='./figure/teaser.png' width='100%' />

## 🖼️ Visual Results

<img src='./figure/compare.png' width='100%' />

## 🔧 Dependencies and Installation

We recommend using Python 3.10 and PyTorch with CUDA support. To set up the environment:

```bash
# Create a new conda environment
conda create -n imagecritic python=3.10
conda activate imagecritic

# Install other dependencies
pip install -r requirements.txt
```

## ⚡ Quick Inference
### Tips
Due to copyright issues, we have embedded the download of the kontext model weights in the inference code below, You can run following inference code directly.
If you have already downloaded the corresponding model, you can comment out the related code and directly replace the inference path.

### Single case inference
```bash
python infer.py
```

### Local Gradio Demo
```bash
python app.py
```

### Single Model Download 
You can download the base model FLUX.1-Kontext-dev directly from [Hugging Face](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev).

Alternatively, you can download it via the following command  
(⚠️ Remember to replace `your_hf_token` in the script with your actual Hugging Face access token):
```bash
python ./download_kontext.py
```



You can download our ImageCritic directly from [Hugging Face](https://huggingface.co/ziheng1234/ImageCritic).

Alternatively, you can download it via following code:

```bash
python ./download_imageCritic.py
```
Or using Git:
```bash
git lfs install
git clone https://huggingface.co/ziheng1234/ImageCritic
```




## Dataset Download 

You can download our training dataset Critic-10K directly from [Hugging Face](https://huggingface.co/datasets/ziheng1234/Critic-10K).

Alternatively, you can download it via Python:

```bash
python /raid/users/oyzh/ImageCritic/download_dataset.py
```
Or using Git:
```bash
git lfs install
git clone https://huggingface.co/datasets/ziheng1234/Critic-10K
```




### Online HuggingFace Demo
You can try ImageCritic demo on [HuggingFace](https://huggingface.co/spaces/ziheng1234/ImageCritic).


##  Citation

If ImageCritic is helpful, please help to ⭐ the repo.

If you find this project useful for your research, please consider citing our [paper]().

## 📧 Contact
If you have any comments or questions, please [open a new issue]() or contact [Ziheng Ouyang](zihengouyang666@gmail.com) 

## License
Licensed under a [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) for Non-commercial use only.
Any commercial use should get formal permission first.
