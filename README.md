# LLM Model Compression & Evaluation Pipeline

This repository contains a Jupyter Notebook that automates the process of compressing Large Language Models (LLMs) and evaluating their performance. The pipeline streamlines the application of quantization techniques and provides a comprehensive evaluation framework.

## 🚀 Overview

The notebook implements a pipeline that covers the following stages:

1.  **Environment Setup**: Installation of necessary libraries.
2.  **Model Selection**: Listing and selection of available LLMs.
3.  **Selective Processing**: Skipping models for compression/evaluation.
4.  **Model Compression**: Applying quantization techniques (INT8, INT4, FP16).
5.  **Model Evaluation**: Evaluating compressed models using relevant metrics.
6.  **Results Reporting**: Generating a comparative evaluation report.

## 🛠️ Key Features

* **Model Compression**:
    * Supports multiple quantization techniques to reduce model size and accelerate inference.
    * Quantization methods include:
        * **INT8 Quantization**: Represents model weights with 8-bit integers.
        * **INT4 Quantization**: Represents model weights with 4-bit integers, offering higher compression.
        * **FP16 Precision**: Represents model weights with 16-bit floating-point numbers.
    * Uses `optimum-intel` for INT8 quantization and `bitsandbytes` for INT4 quantization (if available).
* **Model Evaluation**:
    * Evaluates models using a range of metrics to assess performance.
    * Evaluation metrics include:
        * **ROUGE Score**: Measures the quality of text summarization.
        * **BLEU Score**: Evaluates the quality of machine-translated text.
        * **Inference Time (Latency)**: Measures the average response time.
    * Calculates additional metrics:
        * CHRF Score
        * Unique n-grams, Entropy, Repeated n-grams
        * Coherence (Cosine Similarity)
* **Model Configuration**:
    * LLM models and configurations are loaded from an external Python file (`llm_config.py`).
* **Selective Processing**:
    * Allows users to skip specific models during compression and evaluation, providing flexibility for iterative testing.

## ⚙️ Setup and Installation

### Prerequisites

Ensure you have Python installed. It's recommended to use a virtual environment.

### Installation Steps

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/GodreignElgin/llm-comparision
    cd llm-comparision
    ```

2.  **Install the required packages:**

    ```bash
    pip install -r requirements.txt # If you have a requirements.txt file

    # If not, install the necessary packages directly:
    pip install rouge-score ipywidgets pyngrok
    pip install -Uq pip
    pip uninstall -q -y optimum optimum-intel
    pip install --pre -Uq "openvino>=2024.2.0" openvino-tokenizers[transformers] --extra-index-url [https://storage.openvinotoolkit.org/simple/wheels/nightly](https://storage.openvinotoolkit.org/simple/wheels/nightly)
    pip install -q --extra-index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu) \
    "git+[https://github.com/huggingface/optimum-intel.git](https://github.com/huggingface/optimum-intel.git)" \
    "nncf==2.14.1" \
    "torch>=2.1" \
    "datasets" \
    "accelerate" \
    "huggingface-hub>=0.26.5" \
    "einops" "transformers>=4.43.1" "transformers_stream_generator" "tiktoken" "bitsandbytes"

    # For macOS:
    if platform.system() == "Darwin":
        pip install -q "numpy<2.0.0"
    ```

3.  **Obtain the configuration file:**

    * The notebook attempts to locate `llm_config.py`.
    * If not found, it downloads the file from the specified URL.

    ```python
    import os
    from pathlib import Path
    import requests
    import shutil

    config_shared_path = Path("./llm_config.py")
    config_dst_path = Path("llm_config.py")

    if not config_dst_path.exists():
        if config_shared_path.exists():
            try:
                os.symlink(config_shared_path, config_dst_path)
            except Exception:
                shutil.copy(config_shared_path, config_dst_path)
        else:
            r = requests.get(
                url="[https://github.com/GodreignElgin/llm-comparision/llm_config.py](https://github.com/GodreignElgin/llm-comparision/llm_config.py)"
            )
            with open("llm_config.py", "w", encoding="utf-8") as f:
                f.write(r.text)
    elif not os.path.islink(config_dst_path):
        print("LLM config will be updated")
        if config_shared_path.exists():
            shutil.copy(config_shared_path, config_dst_path)
        else:
            r = requests.get(
                url="[https://github.com/GodreignElgin/llm-comparision/llm_config.py](https://github.com/GodreignElgin/llm-comparision/llm_config.py)"
            )
            with open("llm_config.py", "w", encoding="utf-8") as f:
                f.write(r.text)
    ```

## 🚀 Usage

1.  **Open the Jupyter Notebook:**

    ```bash
    jupyter notebook <notebook_name>.ipynb
    ```

2.  **Follow the notebook's instructions:**

    * The notebook provides interactive widgets to select models for skipping compression or evaluation.
    * It iterates through the models, applies compression, and performs evaluation.
    * Evaluation results are saved in CSV files.

## 🧠 Background Theory

### Model Compression

Model compression techniques aim to reduce the size of machine learning models, making them more efficient for storage and inference. Quantization is a key compression technique used in this pipeline.

* **Quantization**:
    * It reduces the precision of the numbers used to represent a model's weights.
    * This can significantly decrease model size and increase inference speed, often with minimal impact on accuracy.
    * The pipeline implements INT8 and INT4 quantization.
    * **INT8 Quantization**: Representing weights with 8 bits.
    * **INT4 Quantization**: Representing weights with 4 bits for higher compression.

### Evaluation Metrics

The pipeline employs various metrics to evaluate the performance of the compressed LLMs.

* **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**:
    * A set of metrics commonly used to evaluate text summarization and machine translation.
    * ROUGE measures the overlap of n-grams, word sequences, and word pairs between the generated text and reference text(s).
    * For more information, refer to this resource: \[https://www.aclweb.org/anthology/W04-1013]
* **BLEU (Bilingual Evaluation Understudy)**:
    * A metric for evaluating the quality of machine-translated text.
    * BLEU measures the similarity between the machine-generated translation and one or more reference translations.
    * For more information, refer to this resource: \[https://aclanthology.org/P02-1040/]
* **Inference Time (Latency)**:
    * Measures the time it takes for a model to generate a response to a given input.
    * Lower latency indicates faster inference.
* **CHRF (Character n-gram F-score)**:
    * Metric for automatic evaluation of machine translation that uses character n-gram precision and recall.
* **Entropy**:
    * In the context of text generation, entropy can measure the randomness or unpredictability of the generated text.
* **Coherence (Cosine Similarity)**:
    * Cosine similarity can be used to measure the similarity between the embeddings of the input text and the generated text, providing a measure of coherence.

## 📂 File Structure
|── llm_config.py  # Model configurations 
├── <notebook_name>.ipynb # Jupyter Notebook
├── ...

## 📝 Notes

* Ensure sufficient RAM (16-32 GB) for evaluating memory-intensive models.
* Model configurations are loaded from `llm_config.py`, which should be present or downloadable.
* The pipeline supports skipping models for compression/evaluation using interactive widgets.
* Evaluation results are stored in CSV files for analysis.

## 📈 Results Visualization

The notebook includes code to visualize the evaluation results using `matplotlib` and `seaborn`. Example visualizations include:

* Latency vs. Throughput Scatter Plot
* Bar charts for ROUGE-L and BLEU scores

## 🤝 Contributing

Contributions to this project are welcome. Please feel free to submit issues or pull requests.

## 📄 License

This project is licensed under the Apache License 2.0.

## 🙏 Acknowledgements

I would like to acknowledge the following individuals for their contributions to this project:

* \Aditya-2505
* \Gowthaam-J

This project utilizes code from the \[Name of the original GitHub repository].
## References

* ROUGE: \[https://www.aclweb.org/anthology/W04-1013]
* BLEU: \[https://aclanthology.org/P02-1040/]
* Quantization: \[Relevant links to quantization theory and libraries like Optimum Intel, bitsandbytes]
