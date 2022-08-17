# Speaker Embedding Enhancement Model

This repository is for Speaker Embedding Enhancement, followed by Voxceleb Trainer of NAVER Clova AI

https://github.com/clovaai/voxceleb_trainer


### Dependencies
```
conda env create -f conda_requirements.txt
```

### Implemented files

files with names "m_" added at front are new files

```
m_gen_embs.py
m_comb_embs.py
m_gen_list.py
m_DataLoader.py
m_network.py
m_trainer.py
m_viewer.py
loss/m_Losses.py
models/m_Models.py
```

### Data preparation

Prepare Voxceleb dataset by using voxceleb trainer data preparation.

- Single-utterance Embeddings

For prepared data, use m_gen_embs.py for generating embeddings.

Modify "DB_PATH", "EMBED_DIR" for using wav directory and target embedding directory.
```
python ./m_gen_embs.py
```

- Multiple Utterance Embeddings

After Single-utterance Embeddings, Generate Multiple-utterance Embeddings by m_comb_embs.py

set "EMBED_DIR" for using single-utterance embedding directory which will also be target directory.
```
python ./m_comb_embs.py
```

### Training examples

- Using Configuration file:
```
python ./m_trainer.py --config configs/MyLinearNetv2_Exp6-4.yaml
```

You can pass individual arguments that are defined in trainSpeakerNet.py by `--{ARG_NAME} {VALUE}`.
Note that the configuration file overrides the arguments passed via command line.

### Implemented loss functions
```
MSE, MAE, CS
MSE + CS                (MSE_CS)
Softmax
MSE + Softmax           (MSE_Softmax)
Triplet + CS            (MyTriplet_CS)
Domain Adaptation + CS  (DA)
Domain Adaptation + MSE (DA_MSE)
```

### Implemented models
```
MyLinearNet
MyLinearNetv2
MyLinearNetv3
```

### Adding new models and loss functions

You can add new models and loss functions to `modes/m_Models.py` and `loss/m_Losses.py` files. See the existing definitions for examples.

### Accelerating training

- Use `--distributed` flag to enable distributed training.

  - GPU indices should be set before training using the command `export CUDA_VISIBLE_DEVICES=0,1,2,3`.

  - If you are running more than one distributed training session, you need to change the `--port` argument.

### Data

data list have 4 types.
```
{NAME}_normal_list.txt
{NAME}_test_list.txt
{NAME}_triplet_list.txt
{NAME}_distribution_list.txt
```

- train_list, valid_list

For normal loss (MSE, CS, MAE, Softmax, Softmax + CS), use normal list. [ID / FILE]
```
id00000 id00000/youtube_key/12345.wav
id00012 id00012/21Uxsk56VDQ/00001.wav
```

For Triplet loss (Triplet + CS), use triplet list. [ID / FILE / OTHER ID]
```
id10272 id10272/dkN2DIBrXqQ/00006.npy id10308
id10272 id10272/dkN2DIBrXqQ/00005.npy id10303
```

For Domain Adaptation loss (DA + CS, DA + MSE), use distribution list. [ID / FILE1 / FILE2 / SAME_CATEGORY_FLAG]
```
id10272 id10272/dkN2DIBrXqQ/00006.npy id10272/dkN2DIBrXqQ/00005.npy 1
id10284 id10284/YN4cTBWM-QE/00005.npy id10284/RNYNkXzY5Hk/00021.npy 0
```

- test_list, test_list_libri

For EER checking, use test list. [SAME_SPEAKER_FLAG / FILE1 / FILE2]
```
1 id10272/dkN2DIBrXqQ/00006.npy id10272/U-K8tabeDcI/00001.npy
0 id10272/dkN2DIBrXqQ/00006.npy id10292/v6MWr5UAZ94/00002.npy
```

Lists can be created by script "m_gen_list.py", change "EMBED_DIR" and "FILENAME".
```
python ./m_gen_list.py 
```

### Citation - From Clova Voxceleb Trainer

Please cite [1] if you make use of the code. Please see [here](References.md) for the full list of methods used in this trainer.

[1] _In defence of metric learning for speaker recognition_
```
@inproceedings{chung2020in,
  title={In defence of metric learning for speaker recognition},
  author={Chung, Joon Son and Huh, Jaesung and Mun, Seongkyu and Lee, Minjae and Heo, Hee Soo and Choe, Soyeon and Ham, Chiheon and Jung, Sunghwan and Lee, Bong-Jin and Han, Icksang},
  booktitle={Proc. Interspeech},
  year={2020}
}
```

[2] _Clova baseline system for the VoxCeleb Speaker Recognition Challenge 2020_
```
@article{heo2020clova,
  title={Clova baseline system for the {VoxCeleb} Speaker Recognition Challenge 2020},
  author={Heo, Hee Soo and Lee, Bong-Jin and Huh, Jaesung and Chung, Joon Son},
  journal={arXiv preprint arXiv:2009.14153},
  year={2020}
}
```

[3] _Pushing the limits of raw waveform speaker recognition_
```
@article{jung2022pushing,
  title={Pushing the limits of raw waveform speaker recognition},
  author={Jung, Jee-weon and Kim, You Jin and Heo, Hee-Soo and Lee, Bong-Jin and Kwon, Youngki and Chung, Joon Son},
  journal={Proc. Interspeech},
  year={2022}
}
```

### License - From Clova Voxceleb Trainer
```
Copyright (c) 2020-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```
