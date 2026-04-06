<h1 align="center"> <a href=''>ISR: Iterative Search and Reasoning for Video-grounded Dialog</a></h1>

> **TL;DR:** This repository provides the implementation of *Uncovering Hidden Connections: Iterative Search and Reasoning for Video-grounded Dialog*, with an additional runnable text-only workflow for quick training and generation.

![](framework.jpg)

## 📝 Overview
ISR is designed for video-grounded dialog generation and consists of a textual encoder, a visual encoder, and a generator:

- The textual encoder iteratively searches dialogue history for key semantic cues.
- The visual encoder extracts visual evidence relevant to the current question.
- The generator produces the final response from multimodal context.

The repository already includes text data under `process/text_data`, so you can first run the project in **text-only mode**. For full multimodal experiments, you need to prepare the visual feature files separately.

## 🔧 Environment Setup
### 1. Create an Environment
Python 3.9 or later is recommended.

```bash
conda create -n isr python=3.9 -y
conda activate isr
```

### 2. Install Dependencies
Install the appropriate PyTorch build for your CUDA or CPU environment first, then install the remaining dependencies.

```bash
# CPU example
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Remaining dependencies
pip install -r requirements.txt
```

If `torch` is already installed in your environment, you can simply run:

```bash
pip install -r requirements.txt
```

## 📦 Data Preparation
### 1. Built-in Data
The repository already includes the following text data, which can be used directly for text-only debugging and execution:

```text
process/text_data/train_set4DSTC7-AVSD.json
process/text_data/valid_set4DSTC7-AVSD.json
process/text_data/test_set4DSTC7-AVSD.json
process/text_data/lbl_undiscloseonly_test_set4DSTC7-AVSD.json
```

### 2. Full Multimodal Data
The text data, extracted video/audio features, and evaluation tools used in the paper are available from:

- Google Drive: <https://drive.google.com/drive/folders/1SlZTySJAk_2tiMG5F8ivxCfOl_OWwd_Q>
- Charades raw videos: <https://prior.allenai.org/projects/charades>

The official test videos can also be downloaded from:

- <https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_vu17_test.tar>
- <https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_vu17_test_480.tar>

Notes:

- `run.sh` uses `fea_type=none` by default, which means text-only mode.
- To enable visual features, prepare feature files in the format `<FeaType>/<ImageID>.pkl` and modify `fea_type`, `fea_dir`, and `fea_file` in `run.sh`.

## 🗂️ File Structure
Before running the code, the project structure should look like this:

```text
.
├── README.md
├── README_case.md
├── framework.jpg
├── requirements.txt
├── path.sh
├── run.sh
├── train.py
├── generate.py
├── data
│   ├── data_handler.py
│   ├── data_utils.py
│   ├── dataset.py
│   └── coref_data.py
├── model
│   ├── decode.py
│   ├── decoder.py
│   ├── encoder.py
│   ├── generator.py
│   ├── label_smoothing.py
│   ├── modules.py
│   ├── mtn.py
│   └── optimize.py
└── process
    ├── ans_length.py
    ├── clip_frame.py
    ├── statistic.py
    ├── stopwords.txt
    ├── coref_text_data
    ├── prototype
    └── text_data
```

## 🚀 Quick Start
### 1. Run the Default Pipeline
`run.sh` has been adjusted to use the built-in text data and run the text-only version on CPU by default.

```bash
bash run.sh
```

This command will:

1. Train the model
2. Generate responses on the test set using the best checkpoint

The default output directory is:

```text
exps/text-only_warmup13000_epochs60_dropout0.2_faster-rcnn_recome_2_new_decoder_seed1_recome_2_hisqcapvideo/
```

### 2. Training Only
```bash
bash run.sh 2
```

### 3. Custom Training Command
If you want to explicitly control the arguments, run `train.py` directly:

```bash
python3 train.py \
  --gpu -1 \
  --fea-type none \
  --train-path "process/<FeaType>/<ImageID>.pkl" \
  --train-set process/text_data/train_set4DSTC7-AVSD.json \
  --valid-path "process/<FeaType>/<ImageID>.pkl" \
  --valid-set process/text_data/valid_set4DSTC7-AVSD.json \
  --num-epochs 1 \
  --batch-size 8 \
  --model exps/demo/mtn \
  --include-caption summary \
  --separate-caption 1 \
  --max-history-length -1 \
  --merge-source 0 \
  --warmup-steps 1000 \
  --d-model 256 \
  --d-ff 1024 \
  --att-h 8 \
  --dropout 0.2 \
  --cut-a 1 \
  --loss-l 1.0 \
  --diff-encoder 1 \
  --diff-embed 0 \
  --auto-encoder-ft query \
  --diff-gen 0
```

Notes:

- `--gpu -1` means CPU execution.
- `--fea-type none` means no visual features are loaded and the text-only fallback is used.
- For initial debugging, setting `--num-epochs` to `1` or `2` is recommended.

## 🎯 Generation
After training, you can generate test responses with:

```bash
python3 generate.py \
  --gpu -1 \
  --test-path "process/<FeaType>_test/<ImageID>.pkl" \
  --test-set process/text_data/test_set4DSTC7-AVSD.json \
  --model-conf exps/demo/mtn.conf \
  --model exps/demo/mtn_best \
  --beam 5 \
  --penalty 1.0 \
  --nbest 5 \
  --output exps/demo/result_test.json \
  --decode-style beam_search \
  --undisclosed-only 1 \
  --labeled-test process/text_data/lbl_undiscloseonly_test_set4DSTC7-AVSD.json
```

The generated results are saved as a JSON file in AVSD submission format.

## 🧪 Common Usage
### 1. CPU Debugging
```bash
python3 train.py --gpu -1 --fea-type none ...
```

### 2. Single-GPU Training
```bash
python3 train.py --gpu 0 --fea-type none ...
```

### 3. Enable Visual Features
Replace `--fea-type none` with the actual feature name, for example:

```bash
python3 train.py --gpu 0 --fea-type faster-rcnn --train-path "feature/<FeaType>/<ImageID>.pkl" ...
```

This requires the corresponding feature files to be prepared in advance.

## 📁 Output Files
After training, the following files will be generated under the model prefix path:

- `mtn.conf`: vocabulary and training configuration
- `mtn_params.txt`: parameter snapshot
- `mtn_train.csv`: training log
- `mtn_trace.csv`: training and validation losses
- `mtn_best.pth.tar`: best checkpoint on the validation set

## 🎓 Citation
If this project is helpful to your research, please cite the paper.
