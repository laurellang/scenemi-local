# SceneMI: Motion In-betweening for Modeling Human-Scene Interactions

<p align="left">
  <a href='https://arxiv.org/abs/2503.16289'>
    <img src='https://img.shields.io/badge/Arxiv-Pdf-A42C25?style=flat&logo=arXiv&logoColor=white'></a>
  <a href='https://inwoohwang.me/SceneMI/'>
    <img src='https://img.shields.io/badge/Project-Page-green?style=flat&logo=Google%20chrome&logoColor=white'></a>
</p>

![teaser_image](https://inwoohwang.me/SceneMI/static/images/teaser.png)


## :hammer_and_wrench: Setup

```bash
# Create environment
conda create -n scenemi python=3.9
conda activate scenemi

# Install dependencies
pip install -r requirements.txt
```

## :file_folder: Dataset and Preparation

1. **Download the TRUMANS dataset** and place it under:
   ```
   dataset/TRUMANS/Data_release/
   ```

2. **Download SMPL-X body models** and place them under:
   ```
   body_models/smplx/
   ```

3. **Preprocess the dataset:**
   ```bash
   python preprocess_dataset.py
   ```

## :rocket: Training

To train the diffusion-based SceneMI model:
```bash
python -m train.train_diffusion_scenemib
```

## :star: Citation
```
@misc{hwang2025scenemimotioninbetweeningmodeling,
    title={SceneMI: Motion In-betweening for Modeling Human-Scene Interactions}, 
    author={Inwoo Hwang and Bing Zhou and Young Min Kim and Jian Wang and Chuan Guo},
    year={2025},
    eprint={2503.16289},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2503.16289}, 
}
```

## :handshake: Acknowledgements

We sincerely thank the open-source projects that our code builds upon and draws inspiration from:  
CondMDI, TRUMANS and MDM.