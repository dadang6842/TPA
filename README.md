# Temporal Pooling in Consumer-Grade Wearable Human Activity Recognition: Robustness Evaluation under Signal Contamination

This repository implements the methodology proposed in the paper "Temporal Pooling in Consumer-Grade Wearable Human Activity Recognition: Robustness Evaluation under Signal Contamination".

## Paper Overview

**Abstract**: In consumer wearable applications, Human Activity
Recognition (HAR) systems frequently encounter abrupt signal
variations caused by rapid behavioral shifts, which can degrade
performance and lead to inconsistent user experience. However,
evaluating model robustness in these transitional phases is
challenging because natural transitions, despite their real-world
representativeness, often suffer from inherent label ambiguity
that precludes the precise ground truth required for structural
benchmarking. To address this, we present a synthetic
transition stress-test framework for rigorous robustness analysis.
This protocol introduces controlled signal contamination
to simulate severe activity mixtures, providing a label-noisefree
diagnostic tool for the quantitative validation of temporal
pooling mechanisms. Using this framework, we evaluate our
proposed Temporal Pooling Attention (TPA) and its hybrid
variant, Gated-TPA. Our experiments characterize the distinct
roles of each pooling strategy: the proposed TPA acts as a
specialized filter that consistently enhances robustness against
intruder signals at contaminated boundaries, whereas standard
Global Average Pooling (GAP) remains a competitive choice for
synchronized, steady-state segments, a finding further supported
by qualitative analysis of real-world transitions. These findings
provide a practical guideline for designing robust consumer HAR
systems, suggesting that adaptive pooling strategies can effectively
contribute to maintaining robustness under the erratic usage
patterns typical of real-world deployments.

## Dataset

-   **UCI-HAR** dataset is available at _https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones_
-   **PAMAP2** dataset is available at _https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring_
-   **WISDM** dataset is available at _https://www.cis.fordham.edu/wisdm/dataset.php_

## Requirements

```
torch==2.8.0
numpy==2.0.2
pandas==2.2.2
scikit-learn==1.6.1
```

To install all required packages:

```
pip install -r requirements.txt
```

## Codebase Overview

-   `models/tpa_modules.py` - Implementation of the core **Temporal Prototype Attention (TPA)** module.
-   `models/cnn_tpa_models.py` - **CNN Backbone Models**. This file defines the `MultiPathCNN` (a multi-scale temporal feature extractor) and three end-to-end models: `GAPModel` (CNN+GAP), `TPAModel` (CNN+TPA), and `GatedTPAModel` (CNN+Gated-TPA). |
-   `models/lstm_tpa_models.py` - **LSTM Backbone Models**. This file defines the `BiLSTMBackbone` (bidirectional LSTM) and the corresponding three end-to-end models built upon the LSTM feature extractor.
-   `models/transformer_tpa_models.py`- **Transformer Backbone Models**. This file defines the `TransformerBackbone` (Transformer Encoder with Positional Encoding) and the corresponding three end-to-end models utilizing the Transformer feature extractor.

## Citing this Repository

If you use this code in your research, please cite:

```
@article{Temporal Pooling in Consumer-Grade Wearable Human Activity Recognition: Robustness Evaluation under Signal Contamination,
  title = {Temporal Pooling in Consumer-Grade Wearable Human Activity Recognition: Robustness Evaluation under Signal Contamination},
  author={Dahyun Kang and Myung-Kyu Yi}
  journal={},
  volume={},
  Issue={},
  pages={},
  year={}
  publisher={}
}
```

## Contact

For questions or issues, please contact:

-   Dahyun Kang : dadang6842@gmail.com

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
