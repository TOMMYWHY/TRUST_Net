# TRUST-Net

**TRUST-Net: Stage-Wise Reliability Modeling for RGB-D Video Salient Object Detection**

**Manuscript status: Under submission to IEEE Transactions on Multimedia (TMM).**

[中文说明](README_zh-CN.md)

## Partial code release

This repository currently provides a limited set of evaluation utilities for the TRUST-Net project. The core model and training components are not included in this preview.

**After acceptance, we will update this repository with the complete implementation, training and inference scripts, configuration files, and detailed training settings.**

| Component | Availability |
|---|---|
| Generic saliency metric utilities | Included |
| Offline evaluation of saved prediction masks | Included |
| Core TRUST-Net model implementation | Planned after acceptance |
| Training and model inference scripts | Planned after acceptance |
| Complete configurations and training details | Planned after acceptance |
| Model checkpoints and datasets | Not included in this partial release |

The current files support evaluation of user-provided masks. They do not train TRUST-Net, produce predictions, or reproduce the manuscript results on their own. No benchmark scores are distributed with this preview.

## Repository contents

```text
.
├── .gitignore
├── README.md
├── README_zh-CN.md
├── LICENSE
├── THIRD_PARTY_NOTICES.md
├── requirements.txt
├── evaluate.py
├── evaluation/
│   ├── __init__.py
│   └── saliency_metric.py
├── docs/
│   └── EVALUATION.md
└── tests/
    └── test_evaluation.py
```

## Evaluate saved masks

The evaluation utilities run on the CPU and do not require PyTorch or a GPU. Use Python 3.10 with the pinned dependencies in `requirements.txt`; this is the tested environment for the preview.

```bash
python -m pip install -r requirements.txt
python evaluate.py --pred-root predictions --gt-root ground_truth --gt-subdir gt --output results/evaluation.json
```

Expected layout:

```text
predictions/<video>/<frame>.png
ground_truth/<video>/gt/<frame>.png
```

Supply your own prediction and ground-truth masks. Video names and frame names must match exactly. The evaluator checks every ground-truth frame in the specified directory and stops on missing or extra predictions.

See [evaluation conventions](docs/EVALUATION.md) for supported image formats, metric definitions, empty-mask handling, and explicit resizing.

## Acknowledgments and license

The generic saliency metric utilities are redistributed from the project's SAM-DAQ code base with the upstream MIT license and attribution retained. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for their provenance.

The [MIT license](LICENSE) applies to the files distributed in this repository. This preview does not distribute model weights, datasets, or the core model implementation.
