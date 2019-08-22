# LXMERT: Learning Cross-Modality Encoder Representations from Transformers

## Introduction
PyTorch code for our EMNLP 2019 paper ["LXMERT: Learning Cross-Modality Encoder Representations from Transformers"](https://arxiv.org/abs/1908.07490).



## Results (with this Github version)

The accuracy achieved by LXMERT with this code base:


| Split            | [VQA](https://visualqa.org/)     | [GQA](https://cs.stanford.edu/people/dorarad/gqa/)     | [NLVR2](http://lil.nlp.cornell.edu/nlvr/)  |
|-----------       |:----:   |:---:    |:------:|
| local validation | 69.90%  | 59.80%  | 74.95% |
| test-dev         | 72.42%  | 60.00%  | 74.45% (test-P) |
| test-standard    | 72.54%  | 60.33%  | 76.18% (test-U) |

All the results in the table are produced exactly with this code base.
Since [VQA](https://evalai.cloudcv.org/web/challenges/challenge-page/163/overview) and [GQA](https://evalai.cloudcv.org/web/challenges/challenge-page/225/overview) test servers only allow limited number of 'test-standard' submissions,
we use our remaining submission entry from [VQA](https://visualqa.org/challenge.html)/[GQA](https://cs.stanford.edu/people/dorarad/gqa/challenge.html) challenges 2019 to get these results.
For [NLVR2](http://lil.nlp.cornell.edu/nlvr/), we only test once on the unpublished test set (test-U).

**Note that the NLVR2 validation result is slightly different from our paper (74.95% vs 74.5%). The 74.50 result comes from our old weight. 
Althought the code and random seeds are all the same, PyTorch GPU execution is non-determistic which leads to this difference. And thus in [fine-tuning](#fine-tune-on-vision-and-language-tasks), we provide the range of results in multiple runs.**


## Pre-trained models
The pre-trained model (870 MB) is available at `http://nlp.cs.unc.edu/data/model_LXRT.pth`, and could be downloaded with command:
```
mkdir -p snap/pretrained 
wget http://nlp.cs.unc.edu/data/model_LXRT.pth -P snap/pretrained
```


If the downloading speed is slow, the pre-trained model could also be downloaded from [other sources](#alternative-dataset-and-features-download-links), 
and please place it at `snap/pretrained/model_LXRT.pth`.

We also provide the instructions to pre-train the model in [pre-training](#pre-training).
It needs 4 GPUs and takes around a week.



## Fine-tune on Vision-and-Language Tasks
We fine-tune our LXMERT pre-trained model on each task with following hyper-parameters:

|Dataset      | Batch Size   | Learning Rate   | Epochs  | Load Answers  |
|---   |:---:|:---:   |:---:|:---:|
|VQA   | 32  | 5e-5   | 4   | Yes |
|GQA   | 32  | 1e-5   | 4   | Yes |
|NLVR2 | 32  | 5e-5   | 4   | No  |

Although the fine-tuning processes are almost the same except for different hyper-parameters,
we provide descriptions for each dataset to take care of all details.

### General 
The code requires **Python 3** and please install the python libs with command:
```
pip install -r requirements.txt
```

By the way, a python3 virtural environments could be set up and run by:
```
virtualenv name_of_environment -p python3
source name_of_environment/bin/activate
```
### VQA
#### Fine-tuning
* Please make sure the LXMERT pre-trained model is either [downloaded](#pre-trained-models) or [pre-trained](#pre-training).

* Download the re-distributed json files for VQA 2.0 dataset. The raw VQA 2.0 dataset could be downloaded from the [official website](https://visualqa.org/download.html).
```
mkdir -p data/vqa
wget nlp.cs.unc.edu/data/lxmert_data/vqa/train.json -P data/vqa/
wget nlp.cs.unc.edu/data/lxmert_data/vqa/nominival.json -P  data/vqa/
wget nlp.cs.unc.edu/data/lxmert_data/vqa/minival.json -P data/vqa/
```
* Download faster-rcnn features for MS COCO train2014 (17 GB) and val2014 (8 GB) images (VQA 2.0 is collected on MS COCO dataset).
The image features are
also available on Google Drive and Baidu Drive (see [Alternative Download](#alternative-dataset-and-features-download-links) for details).
```
mkdir -p data/mscoco_imgfeat
wget nlp.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/train2014_obj36.zip -P data/mscoco_imgfeat
unzip data/mscoco_imgfeat/train2014_obj36.zip -d data/mscoco_imgfeat && rm data/mscoco_imgfeat/train2014_obj36.zip
wget nlp.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/val2014_obj36.zip -P data/mscoco_imgfeat
unzip data/mscoco_imgfeat/val2014_obj36.zip -d data && rm data/mscoco_imgfeat/val2014_obj36.zip
```

* Before fine-tuning on whole VQA 2.0 training set, verifying the script and model on a small training set (512 images) is recommended. 
The first argument `0` is GPU id. The second argument `vqa_lxr955_tiny` is the name of this experiment.
```
bash run/vqa_finetune.bash 0 vqa_lxr955_tiny --tiny
```
* If no bug came out, then the model is ready to be trained on the whole VQA corpus:
```
bash run/vqa_finetune.bash 0 vqa_lxr955
```
* It takes around 8 hours (2 hours per epoch * 4 epochs) to converge. 
The **logs** and **model snapshots** will be saved under folder `snap/vqa/vqa_lxr955`. 
The validation result after training will be around **69.7%** to **70.2%**. 

#### Local Validation
The results on the validation set (our minival set) are printed while training.
The validation result is also saved to `snap/vqa/[experiment-name]/log.log`.
If the log file was accidentally deleted, the validation result in training is also reproducible from the model snapshot:
```
bash run/vqa_test.bash 0 vqa_lxr955_results --test minival --load snap/vqa/vqa_lxr955/BEST
```
#### Submitted to VQA test server
- Download our re-distributed json file containing VQA 2.0 test data.
```
wget nlp.cs.unc.edu/data/lxmert_data/vqa/test.json -P data/vqa/
```
- Download the faster rcnn features for MS COCO test2015 split (16 GB).
```
wget nlp.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/test2015_obj36.zip -P data/mscoco_imgfeat
unzip data/mscoco_imgfeat/test2015_obj36.zip -d data && rm data/mscoco_imgfeat/test2015_obj36.zip
```
- Since VQA submission system requires submitting whole test data, we need to run inference over all test splits 
(i.e., test dev, test standard, test challenge, and test held-out). 
It takes around 10~15 mins to run test inference (448K instances to run).
```
bash run/vqa_test.bash 0 vqa_lxr955_results --test test --load snap/vqa/vqa_lxr955/BEST
```
- The test results will be saved in `snap/vqa_lxr955_results/test_predict.json`. 
The VQA 2.0 challenge for this year is host on [EvalAI](https://evalai.cloudcv.org/) at [https://evalai.cloudcv.org/web/challenges/challenge-page/163/overview](https://evalai.cloudcv.org/web/challenges/challenge-page/163/overview)
It still allows submission after the challenge ended.
Please check the official webiste of [VQA Challenge](https://visualqa.org/challenge.html) for detailed information and 
follow the instructions in [EvalAI](https://evalai.cloudcv.org/web/challenges/challenge-page/163/overview) to submit.
In general, after registration, the only thing remaining is to upload the `test_predict.json` file and wait for the result back.

- The testing accuracy with exact this code is **72.42%** for test-dev and **72.54%**  for test-standard.
The results with the code base are also publicly shown on the [VQA 2.0 leaderboard](
https://evalai.cloudcv.org/web/challenges/challenge-page/163/leaderboard/498
) with entry `LXMERT github version`.


### GQA

#### Fine-tuning
* Please make sure the LXMERT pre-trained model is either [downloaded](#pre-trained-models) or [pre-trained](#pre-training).

* Download the re-distributed json files for GQA balanced version dataset.
The original GQA dataset is available [here](https://cs.stanford.edu/people/dorarad/gqa/download.html) and the script to 
preprocess these datasets is under `data/gqa/process_raw_data_scripts`.
```
mkdir -p data/gqa
wget nlp.cs.unc.edu/data/lxmert_data/gqa/train.json -P data/gqa/
wget nlp.cs.unc.edu/data/lxmert_data/gqa/valid.json -P data/gqa/
wget nlp.cs.unc.edu/data/lxmert_data/gqa/testdev.json -P data/gqa/
```
* Download faster-rcnn features for Vsiual Genome and GQA images (30 GB).
GQA's training and validation data are collected from Visual Genome.
Its testing images come from MS COCO test set (I have verified this with GQA author [Drew A. Hudson](https://www.linkedin.com/in/drew-a-hudson/)).
The image features are
also available on Google Drive and Baidu Drive (see [Alternative Download](#alternative-dataset-and-features-download-links) for details).
```
mkdir -p data/vg_gqa_imgfeat
wget nlp.cs.unc.edu/data/lxmert_data/vg_gqa_imgfeat/vg_gqa_obj36.zip -P data/vg_gqa_imgfeat
unzip data/vg_gqa_imgfeat/vg_gqa_obj36.zip -d data && rm data/vg_gqa_imgfeat/vg_gqa_obj36.zip
wget nlp.cs.unc.edu/data/lxmert_data/vg_gqa_imgfeat/gqa_testdev_obj36.zip -P data/vg_gqa_imgfeat
unzip data/vg_gqa_imgfeat/gqa_testdev_obj36.zip -d data && rm data/vg_gqa_imgfeat/gqa_testdev_obj36.zip
```

* Before fine-tuning on whole GQA training+validation set, verifying the script and model on a small training set (512 images) is recommended. 
The first argument `0` is GPU id. The second argument `gqa_lxr955_tiny` is the name of this experiment.
```
bash run/gqa_finetune.bash 0 gqa_lxr955_tiny --tiny
```

* If no bug came out, then the model is ready to be trained on the whole GQA corpus (train + validation), and validate on 
the testdev set:
```
bash run/gqa_finetune.bash 0 gqa_lxr955
```
* It takes around 16 hours (4 hours per epoch * 4 epochs) to converge. 
The **logs** and **model snapshots** will be saved under folder `snap/gqa/gqa_lxr955`. 
The validation result after training will be around **59.8%** to **60.1%**. 

#### Local Validation
The results on testdev is printed out while training and saved in `snap/gqa/gqa_lxr955/log.log`.
It could be also re-calculated with command:
```
bash run/gqa_test.bash 0 gqa_lxr955_results --load snap/gqa/gqa_lxr955/BEST --test testdev --batchSize 1024
```

> Note: Our local testdev result is usually 0.1% to 0.5% lower than the 
submitted testdev result. 
The reason is that the test server takes an [advanced 
evaluation system](https://cs.stanford.edu/people/dorarad/gqa/evaluate.html) while our local evaluator only 
calculates the exact matching.
Please use [this official evaluator](https://nlp.stanford.edu/data/gqa/eval.zip) if you 
want to have the exact number without submitting.


#### Submitted to GQA test server
- Download our re-distributed json file containing GQA test data.
```
wget nlp.cs.unc.edu/data/lxmert_data/gqa/submit.json -P data/gqa/
```

- Since GQA submission system requires submitting the whole test data, 
we need to run inference over all test splits.
It takes around 30~60 mins to run test inference (4.2M instances to run).
```
bash run/gqa_test.bash 0 gqa_lxr955_results --load snap/gqa/gqa_lxr955/BEST --test submit --batchSize 1024
```

- After running test script, a json file `submit_predict.json` under `snap/gqa/gqa_lxr955_results` will contain 
all the prediction results and is ready to be submitted.
The GQA challenge 2019 is hosted by [EvalAI](https://evalai.cloudcv.org/) at [https://evalai.cloudcv.org/web/challenges/challenge-page/225/overview](https://evalai.cloudcv.org/web/challenges/challenge-page/225/overview).
After registering the account, uploading the `submit_predict.json` and waiting for the results are the only thing remained.
Please also check [GQA official website](https://cs.stanford.edu/people/dorarad/gqa/) 
in case the test server is changed.

- The testing accuracy with exact this code is **60.00%** for test-dev and **60.33%**  for test-standard.
The results with the code base are also publicly shown on the [GQA leaderboard](
https://evalai.cloudcv.org/web/challenges/challenge-page/225/leaderboard
) with entry `LXMERT github version`.

### NLVR2

#### Fine-tuning

- Download the NLVR2 data from the official [github repo](https://github.com/lil-lab/nlvr).
```
git clone https://github.com/lil-lab/nlvr.git data/nlvr2/nlvr
```

- Process the NLVR2 data to json files.
```
bash -c "cd data/nlvr2/process_raw_data_scripts && python process_dataset.py"
```

- Download the NLVR2 image features for train (21 GB) & valid (1.6 GB) splits. 
The image features are
also available on Google Drive and Baidu Drive (see [Alternative Download](#alternative-dataset-and-features-download-links) for details).
To access to the original images, please follow the instructions on [NLVR2 official Github](https://github.com/lil-lab/nlvr/tree/master/nlvr2).
The images could either be downloaded with the urls or by signing an agreement form for data usage. And the feature could be extracted as described in [feature extraction](#faster-r-cnn-feature-extraction)
```
mkdir -p data/nlvr2_imgfeat
wget nlp.cs.unc.edu/data/lxmert_data/nlvr2_imgfeat/train_obj36.zip -P data/nlvr2_imgfeat
unzip data/nlvr2_imgfeat/train_obj36.zip -d data && rm data/nlvr2_imgfeat/train_obj36.zip
wget nlp.cs.unc.edu/data/lxmert_data/nlvr2_imgfeat/valid_obj36.zip -P data/nlvr2_imgfeat
unzip data/nlvr2_imgfeat/valid_obj36.zip -d data && rm data/nlvr2_imgfeat/valid_obj36.zip
```

- Before fine-tuning on whole NLVR2 training set, verifying the script and model on a small training set (512 images) is recommended. 
The first argument `0` is GPU id. The second argument `nlvr2_lxr955_tiny` is the name of this experiment.
Do not worry if the result is low (50~55) on this tiny split, 
the whole training data would bring the performance back.
```
bash run/nlvr2_finetune.bash 0 nlvr2_lxr955_tiny --tiny
```

- If no bugs popping up in previous step, 
it means that the code, the data, and image features are ready.
Please use this command to train on the full training set. 
The result on NLVR2 validation (dev) set would be around **74.0** to **74.5**.
```
bash run/nlvr2_finetune.bash 0 nlvr2_lxr955
```

#### Inference on Public Test Split
- Download NLVR2 image features for the public test split (1.6 GB).
```
wget nlp.cs.unc.edu/data/lxmert_data/nlvr2_imgfeat/test_obj36.zip -P data/nlvr2_imgfeat
unzip data/nlvr2_imgfeat/test_obj36.zip -d data/nlvr2_imgfeat && rm data/nlvr2_imgfeat/test_obj36.zip
```

- Test on the public test set (corresponding to 'test-P' on [NLVR2 leaderboard](http://lil.nlp.cornell.edu/nlvr/)) with:
```
bash run/nlvr2_test.bash 0 nlvr2_lxr955_results --load snap/nlvr2/nlvr2_lxr955/BEST --test test --batchSize 1024
```

- The test accuracy would be shown on the screen after around 5~10 minutes.
It also saves the predictions in the file `test_predict.csv` 
under `snap/nlvr2_lxr955_reuslts`, which is compatible to NLVR2 [official evaluation script](https://github.com/lil-lab/nlvr/tree/master/nlvr2/eval).
The official eval script also calculates consistency ('Cons') besides the accuracy.
We could use this official script to verify the results by running:
```
python data/nlvr2/nlvr/nlvr2/eval/metrics.py snap/nlvr2/nlvr2_lxr955_results/test_predict.csv data/nlvr2/nlvr/nlvr2/data/test1.json
```

- The accuracy of public test ('test-P') set should be almost same to the validation set ('dev'),
which is around 74.0% to 74.5%.


#### Unreleased Test Sets
To be tested on the unreleased held-out test set (test-U on the 
[leaderboard](http://lil.nlp.cornell.edu/nlvr/)
),
the code needs to be sent.
Please check the [NLVR2 official github](https://github.com/lil-lab/nlvr/tree/master/nlvr2) 
and [NLVR project website](http://lil.nlp.cornell.edu/nlvr/) for details.


### General Debugging Options
Since it takes a few minutes to load the features, the code has an option to prototype with a small amount of
training data. 
```
# Training with 512 images:
bash run/vqa_finetune.bash 0 --tiny 
# Training with 4096 images:
bash run/vqa_finetune.bash 0 --fast
```

## Pre-training

- Download the our aggregated LXMERT dataset (around 700MB in total)
```
mkdir -p data/lxmert
wget nlp.cs.unc.edu/data/lxmert_data/lxmert/mscoco_train.json -P data/lxmert/
wget nlp.cs.unc.edu/data/lxmert_data/lxmert/mscoco_nominival.json -P data/lxmert/
wget nlp.cs.unc.edu/data/lxmert_data/lxmert/vgnococo.json -P data/lxmert/
wget nlp.cs.unc.edu/data/lxmert_data/lxmert/mscoco_minival.json -P data/lxmert/
```

- *Skip this if you have run [VQA fine-tuning](#vqa).* Download the detection features for MS COCO images.
```
mkdir -p data/mscoco_imgfeat
wget nlp.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/train2014_obj36.zip -P data/mscoco_imgfeat
unzip data/mscoco_imgfeat/train2014_obj36.zip -d data/mscoco_imgfeat && rm data/mscoco_imgfeat/train2014_obj36.zip
wget nlp.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/val2014_obj36.zip -P data/mscoco_imgfeat
unzip data/mscoco_imgfeat/val2014_obj36.zip -d data && rm data/mscoco_imgfeat/val2014_obj36.zip
```

- *Skip this if you have run [GQA fine-tuning](#gqa).* Download the detection features for Visual Genome images.
```
mkdir -p data/vg_gqa_imgfeat
wget nlp.cs.unc.edu/data/lxmert_data/vg_gqa_imgfeat/vg_gqa_obj36.zip -P data/vg_gqa_imgfeat
unzip data/vg_gqa_imgfeat/vg_gqa_obj36.zip -d data && rm data/vg_gqa_imgfeat/vg_gqa_obj36.zip
```

- Test on a small split of the MS COCO + Visual Genome datasets:
```
bash run/lxmert_pretrain.bash 0,1,2,3 --multiGPU --tiny
```

- Run on the whole MS COCO + Visual Genome datasets. Here, we take a simple one-step pre-training strategy rather than the two-steps (10 epochs without image QA and 10 epochs with image QA) methods describe in our paper. We re-run the pre-training and did not find much difference with these two strategies.
```
bash run/lxmert_pretrain.bash 0,1,2,3 --multiGPU
```

- Explanation of arguments in the pre-training script `run/lxmert_pretrain.bash`:
```
python src/pretrain/lxmert_pretrain_new.py \
    # The pre-training tasks
    --taskMaskLM --taskObjPredict --taskMatched --taskQA \  
    
    # Vision subtasks
    # obj / attr: detected object/attribute label prediction.
    # feat: RoI feature regression.
   	 --visualLosses obj,attr,feat \
    
    # Mask rate for words and objects
    --wordMaskRate 0.15 --objMaskRate 0.15 \
    
    # Training and validation sets
    # mscoco_nominival + mscoco_minival = mscoco_val2014
    # visual genome - mscoco = vgnococo
    --train mscoco_train,mscoco_nominival,vgnococo --valid mscoco_minival \
    
    # Number of layers in each encoder
    --llayers 9 --xlayers 5 --rlayers 5 \
    
    # Train from scratch (Using intialized weights) instead of loading BERT weights.
    --fromScratch \
    # Hyper parameters
    --batchSize 256 --optim bert --lr 1e-4 --epochs 12 \
    --tqdm --output $output ${@:2}
```


## Alternative Dataset and Features Download Links 
All default download links are provided by our servers in [UNC CS department](https://cs.unc.edu) and under 
our [NLP group website](https://nlp.cs.unc.edu) but the network bandwidth might be limited. 
We thus provide a few other options with Google Drive and Baidu Drive.

The files in online drives are almost structured in the same way 
as our repo but have a few differences due to specific policies.
After downloading the data and features from the drives, 
please re-organize them under `data/` folder according to the following example:
```
REPO ROOT
 |
 |-- data                  
 |    |-- vqa
 |    |    |-- train.json
 |    |    |-- minival.json
 |    |    |-- nominival.json
 |    |    |-- test.json
 |    |
 |    |-- mscoco_imgfeat
 |    |    |-- train2014_obj36.tsv
 |    |    |-- val2014_obj36.tsv
 |    |    |-- test2015_obj36.tsv
 |    |
 |    |-- vg_gqa_imgfeat -- *.tsv
 |    |-- gqa -- *.json
 |    |-- nlvr2_imgfeat -- *.tsv
 |    |-- nlvr2 -- *.json
 |    |-- lxmert -- *.json          # Pre-training data
 | 
 |-- snap
 |-- src
```

Please also kindly contact us if anything is missing!

### Google Drive
As an alternative way to download feature from our UNC server,
you could also download the feature from google drive with link [https://drive.google.com/drive/folders/1Gq1uLUk6NdD0CcJOptXjxE6ssY5XAuat?usp=sharing](https://drive.google.com/drive/folders/1Gq1uLUk6NdD0CcJOptXjxE6ssY5XAuat?usp=sharing).
The structure of the folders on drive is:
```
Google Drive Root
 |-- data                  # The raw data and image features without compression
 |    |-- vqa
 |    |-- gqa
 |    |-- mscoco_imgfeat
 |    |-- ......
 |
 |-- image_feature_zips    # The image-feature zip files (Around 45% compressed)
 |    |-- mscoco_imgfeat.zip
 |    |-- nlvr2_imgfeat.zip
 |    |-- vg_gqa_imgfeat.zip
 |
 |-- snap -- pretrained -- model_LXRT.pth # The pytorch pre-trained model weights.
```
Note: image features in zip files (e.g., `mscoco_mgfeat.zip`) are the same to which in `data/` (i.e., `data/mscoco_imgfeat`). 
If you want to save network bandwidth, please download the feature zips and skip downloading the `*_imgfeat` folders under `data/`.
### Baidu Drive

Since [Google Drive](
https://drive.google.com/drive/folders/1Gq1uLUk6NdD0CcJOptXjxE6ssY5XAuat?usp=sharing
) is not officially available across the world,
we also create a mirror on Baidu drive (i.e., Baidu PAN). 
The dataset and features could be downloaded with shared link 
[https://pan.baidu.com/s/1m0mUVsq30rO6F1slxPZNHA](https://pan.baidu.com/s/1m0mUVsq30rO6F1slxPZNHA) 
and access code `wwma`.
```
Baidu Drive Root
 |
 |-- vqa
 |    |-- train.json
 |    |-- minival.json
 |    |-- nominival.json
 |    |-- test.json
 |
 |-- mscoco_imgfeat
 |    |-- train2014_obj36.zip
 |    |-- val2014_obj36.zip
 |    |-- test2015_obj36.zip
 |
 |-- vg_gqa_imgfeat -- *.zip.*  # Please read README.txt under this folder
 |-- gqa -- *.json
 |-- nlvr2_imgfeat -- *.zip.*   # Please read README.txt under this folder
 |-- nlvr2 -- *.json
 |-- lxmert -- *.json
 | 
 |-- pretrained -- model_LXRT.pth
```

Since Baidu Drive does not support extremely large files, 
we `split` a few features zips into multiple small files. 
Please follow the `README.txt` under `baidu_drive/vg_gqa_imgfeat` and 
`baidu_drive/nlvr2_imgfeat` to concatenate back to the feature zips with command `cat`.


## Code and Project Explanation
- All code is in folder `src`. The basic `lxrt` .
The python files related to pre-training and fine-tuning are saved in `src/pretrain` and `src/tasks` respectively.
- I kept folders containing image features (e.g., mscoco_imgfeat) separated from vision-and-language dataset (e.g., vqa, lxmert) because
multiple vision-and-language datasets would share common images.
- We use the name `lxmert` for our framework and use the name `lxrt`
(Language, Cross-Modality, and object-Relationship Transformers) to refer to our our models.
- Similar to `lxrt` ((Language, Cross-Modality, and object-Relationship Transformers), 
we use `lxr???` to annotate the number of layers in each components.
E.g., `lxr955` (as in this code base) refers to 
a model with 9 Language layers, 5 cross-modality layers, and 5 object-Relationship layers.

## Faster R-CNN Feature Extraction


We use the Faster R-CNN feature extractor demonstrated in ["Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering", CVPR 2018](https://arxiv.org/abs/1707.07998)
and its code released at [bottom-up-attention github repo](https://github.com/peteanderson80/bottom-up-attention).
It was trained on [Visual Genome](https://visualgenome.org/) dataset and implemented based on a specific [Caffe](https://caffe.berkeleyvision.org/) version.
To extract object features, you could follow the installation instructions in the bottom-up attention github [https://github.com/peteanderson80/bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention). 

We also provide a docker image which takes care of all these dependencies.

### Feature Extraction with Docker
[Docker](https://www.docker.com/) is a easy-to-use virturlization tool which allows you to plug and play without installing libraries.

The built docker file for bottom-up-attention is released on [docker hub](https://hub.docker.com/r/airsplay/bottom-up-attention) and could be downloaded with command:
```
sudo docker pull airsplay/bottom-up-attention
```
After pulling the docker, you could test running the docker container with command:
```
docker run --gpus all --rm -it airsplay/bottom-up-attention bash
``` 

If erros about `--gpus all` popped up, please read the next section.

#### Docker GPU Access
Note that the purpose of the argument `--gpus all` is to expose GPU devices to the docker container, and it requires `docker` version >= 19.03 with latest `nvidia-docker` support.
The two requirements could be installed following the instructions on the website:
1. Docker CE (19.03): https://docs.docker.com/v17.09/engine/installation/linux/docker-ce/ubuntu/
2. nvidia-docker: https://github.com/NVIDIA/nvidia-docker

For docker with old version, either updating it to docker 19.03 or using command `--runtime=nvidia` instead of `--gpus all' should help.

#### An Example: Feature Extraction for NLVR2 
We demonstrate how to extract Faster R-CNN features of NLVR2 images.

- Please first following the instruction on [NLVR2 official github](https://github.com/lil-lab/nlvr/tree/master/nlvr2) to get the images.

- Download the pre-trained Faster R-CNN model. Instead of using the default pre-trained model (trained with 10 to 100 boxes), we use the ['alternative pretrained model'](https://github.com/peteanderson80/bottom-up-attention#demo) which trained with 36 boxes. 
```
wget https://www.dropbox.com/s/bacig173qnxddvz/resnet101_faster_rcnn_final_iter_320000.caffemodel?dl=1 -O data/nlvr2_imgfeat/resnet101_faster_rcnn_final_iter_320000.caffemodel
```

- Run docker container with command:
```
docker run --gpus all -v /path/to/nlvr2/images:/workspace/images:ro -v /path/to/lxrt_public/data/nlvr2_imgfeat:/workspace/features --rm -it airsplay/bottom-up-attention bash
```
`-v` mounts the folders on host os to the docker image container.
> Note0: If it says something about 'privilege', add `sudo` before the command.
>
> Note1: If it says something about '--gpus all', it means that the GPU options are not correctly set. Please read [Docker GPU Access](#docker-gpu-access) for the instructions to allow GPU access.
>
> Note2: /path/to/nlvr2/images would contain subfolders `train`, `dev`, `test1` and `test2`.
>
> Note3: Both paths '/path/to/nlvr2/images/' and '/path/to/lxrt_public' requires absolute paths.


- Extract the features **inside the docker container**. The extraction script is copied from 
```
cd /workspace/features
CUDA_VISIBLE_DEVICES=0 python extract_nlvr2_image.py --split train 
CUDA_VISIBLE_DEVICES=0 python extract_nlvr2_image.py --split valid
CUDA_VISIBLE_DEVICES=0 python extract_nlvr2_image.py --split test
```

- It would takes around 5 to 6 hours for the training split and 1 to 2 hours for the valid and test splits. Since it is slow, I recommend to run them parallelly if there are multiple GPUs. It could be achived by changing the `gpu_id` in `CUDA_VISIBLE_DEVICES=$gpu_id`.

- The features would be saved in `train.tsv`, `valid.tsv`, and `test.tsv` under dir `data/nlvr2_imgfeat` outside the docker container. I have verified the extracted image features are the same to the one I provided in [NLVR2 fine-tuning](#nlvr2).


## Reference
If you find this project helps, please cite our paper :)
```
@inproceedings{tan2019lxmert,
  title={LXMERT: Learning Cross-Modality Encoder Representations from Transformers},
  author={Tan, Hao and Bansal, Mohit},
  booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing},
  year={2019}
}
```

## Acknowledgement
We thank the funding support from ARO-YIP Award #W911NF-18-1-0336, & awards from Google, Facebook, Salesforce, and Adobe.

We thank [Peter Anderson](https://panderson.me/) for providing the faster R-CNN code and pre-trained models under
[Bottom-Up-Attention Github Repo](https://github.com/peteanderson80/bottom-up-attention).

We thank [hugginface](https://github.com/huggingface) for releasing the excellent PyTorch code 
[PyTorch Transformers](https://github.com/huggingface/pytorch-transformers).

We thank [Hengyuan Hu](https://www.linkedin.com/in/hengyuan-hu-8963b313b) for his [PyTorch VQA](https://github.com/hengyuan-hu/bottom-up-attention-vqa) implementation, our local VQA evaluator borrows the idea from this repo.

We thank [Alane Suhr](http://alanesuhr.com/) for helping test LXMERT on NLVR2 unreleased test split.

We thank all the authors and annotators of vision-and-language datasets 
(i.e., 
[MS COCO](http://cocodataset.org/#home), 
[Visual Genome](https://visualgenome.org/),
[VQA](https://visualqa.org/),
[GQA](https://cs.stanford.edu/people/dorarad/gqa/),
[NLVR2](http://lil.nlp.cornell.edu/nlvr/)
), 
which allows us to develop a pre-trained model for vision-and-language tasks.

We thank [Jie Lei](http://www.cs.unc.edu/~jielei/) and [Licheng Yu](http://www.cs.unc.edu/~licheng/) for their helpful discussions. I also want to thank [Shaoqing Ren](https://www.shaoqingren.com/) to teach me vision knowledge when I was in MSRA.

We also thank you to look into our code. Please kindly contact us if you find any issue. Comments are always welcome.

LXRThanks.
