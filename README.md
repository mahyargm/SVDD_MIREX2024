# MIREX 2024: Singing Voice DeepFake Detection (SVDD) Challenge - Winning Solution

## Overview

This repository contains the codebase for our **first-place submission** to the [Singing Voice DeepFake Detection (SVDD) challenge](https://www.music-ir.org/mirex/wiki/2024:Singing_Voice_Deepfake_Detection) in the [MIREX](https://www.music-ir.org/mirex/wiki/MIREX_HOME) track at [ISMIR 2024](https://ismir2024.ismir.net/). Our approach leverages **ResNet18** as the backbone model to predict the authenticity scores of audio segments, using their **log-spectrogram representations**.

Key Highlights:
- **Model Architecture**: Utilizes ResNet18 with specialized training for both mixed songs and isolated vocals.
- **Detailed Implementation**: For comprehensive details, refer to our [system description](System_Description.pdf).
- **Dataset**: The **WildSVDD dataset** served as the primary dataset for this research. For more details, visit the SVDD challenge's [homepage](https://www.music-ir.org/mirex/wiki/2024:Singing_Voice_Deepfake_Detection).

## Installation

Ensure Python 3.x is installed and run the following command to install dependencies:

```bash
pip install -r requirements.txt
```

## Data Preprocessing:

All input audio segments must be of equal length (4 seconds) to train or evaluate the model. To preprocess and segment your dataset, use preprocess.py:

```bash
python preprocess.py <dataset_directory> <mixture/vocals partition> <output_directory>
```
## Training:
To train the model, run the Training.py script:
```bash
python Training.py <dataset_directory> <mixture/vocals partition> <experiment_name>
```

## Evaluation:
To evaluate the trained model, use Test.py:
```bash
python Test.py <dataset_directory> <mixture/vocals partition> <experiment_name>
```
## Pretrained Models
Download pretrained models for reproducibility or further experiments:

- Model trained on mixed songs: [Download here](https://drive.google.com/file/d/1foUR3r_UGQvWho6V-V9oq_-gZ3cijvQY/view?usp=sharing)
- Model trained on isolated vocals: [Download here](https://drive.google.com/file/d/1bByfTT5pcMAPk4Y4HU8YrAFkCDu_zmHK/view?usp=sharing)
## Acknowledgments:

We extend our sincere gratitude to the SVDD Challenge organizers for curating the WildSVDD dataset, which played a crucial role in enabling this research and contributing to advancements in audio forensics.
