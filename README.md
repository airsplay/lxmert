# LXMERT: Learning Cross-Modality Encoder Representations from Transformers
**Important: Due to a URL earthquake happend in our group, the previous links needs to be replaced. I tried my best to save every lifes in this document. If you find any link is not available, please do not hesitate to contact me. Thanks. **
## Introduction
PyTorch code for the EMNLP 2019 paper ["LXMERT: Learning Cross-Modality Encoder Representations from Transformers"](https://arxiv.org/abs/1908.07490). Slides of our EMNLP 2019 talk are avialable [here](http://www.cs.unc.edu/~airsplay/EMNLP_2019_LXMERT_slides.pdf). 

- To analyze the output of pre-trained model (instead of fine-tuning on downstreaming tasks), please load the weight `https://nlp1.cs.unc.edu/data/github_pretrain/lxmert20/Epoch20_LXRT.pth`, which is trained as in section [pre-training](#pre-training). The default weight [here](#pre-trained-models) is trained with a slightly different protocal as this code.


## Results (with this Github version)

| Split            | [VQA](https://visualqa.org/)     | [GQA](https://cs.stanford.edu/people/dorarad/gqa/)     | [NLVR2](http://lil.nlp.cornell.edu/nlvr/)  |
|-----------       |:----:   |:---:    |:------:|
| Local Validation | 69.90%  | 59.80%  | 74.95% |
| Test-Dev         | 72.42%  | 60.00%  | 74.45% (Test-P) |
| Test-Standard    | 72.54%  | 60.33%  | 76.18% (Test-U) |

All the results in the table are produced exactly with this code base.
Since [VQA](https://evalai.cloudcv.org/web/challenges/challenge-page/163/overview) and [GQA](https://evalai.cloudcv.org/web/challenges/challenge-page/225/overview) test servers only allow limited number of 'Test-Standard' submissions,
we use our remaining submission entry from the [VQA](https://visualqa.org/challenge.html)/[GQA](https://cs.stanford.edu/people/dorarad/gqa/challenge.html) challenges 2019 to get these results.
For [NLVR2](http://lil.nlp.cornell.edu/nlvr/), we only test once on the unpublished test set (test-U).

We use this code (with model ensemble) to participate in [VQA 2019](https://visualqa.org/roe.html) and [GQA 2019](https://drive.google.com/open?id=1CtFk0ldbN5w2qhwvfKrNzAFEj-I9Tjgy) challenge in May 2019.
We are the **only** team ranking **top-3** in both challenges.


## Pre-trained models
The pre-trained model (870 MB) is available at http://nlp1.cs.unc.edu/data/model_LXRT.pth, and can be downloaded with:
```bash
mkdir -p snap/pretrained 
wget --no-check-certificate http://nlp1.cs.unc.edu/data/model_LXRT.pth -P snap/pretrained
```


If download speed is slower than expected, the pre-trained model could also be downloaded from [other sources](#alternative-dataset-and-features-download-links).
Please help put the downloaded file at `snap/pretrained/model_LXRT.pth`.

We also provide data and commands to pre-train the model in [pre-training](#pre-training). The default setup needs 4 GPUs and takes around a week to finish. The pre-trained weights with this code base could be downloaded from `https://nlp1.cs.unc.edu/data/github_pretrain/lxmert/EpochXX_LXRT.pth`, `XX` from 01 to 12. It is pre-trained for 12 epochs (instead of 20 in EMNLP paper) thus the fine-tuned reuslts are about 0.3% lower on each datasets. 



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
The code requires **Python 3** and please install the Python dependencies with the command:
```bash
pip install -r requirements.txt
```

By the way, a Python 3 virtual environment could be set up and run with:
```bash
virtualenv name_of_environment -p python3
source name_of_environment/bin/activate
```
### VQA
#### Fine-tuning
1. Please make sure the LXMERT pre-trained model is either [downloaded](#pre-trained-models) or [pre-trained](#pre-training).

2. Download the re-distributed json files for VQA 2.0 dataset. The raw VQA 2.0 dataset could be downloaded from the [official website](https://visualqa.org/download.html).
    ```bash
    mkdir -p data/vqa
    wget --no-check-certificate https://nlp1.cs.unc.edu/data/lxmert_data/vqa/train.json -P data/vqa/
    wget --no-check-certificate https://nlp1.cs.unc.edu/data/lxmert_data/vqa/nominival.json -P  data/vqa/
    wget --no-check-certificate https://nlp1.cs.unc.edu/data/lxmert_data/vqa/minival.json -P data/vqa/
    ```
3. Download faster-rcnn features for MS COCO train2014 (17 GB) and val2014 (8 GB) images (VQA 2.0 is collected on MS COCO dataset).
The image features are
also available on Google Drive and Baidu Drive (see [Alternative Download](#alternative-dataset-and-features-download-links) for details).
    ```bash
    mkdir -p data/mscoco_imgfeat
    wget --no-check-certificate https://nlp1.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/train2014_obj36.zip -P data/mscoco_imgfeat
    unzip data/mscoco_imgfeat/train2014_obj36.zip -d data/mscoco_imgfeat && rm data/mscoco_imgfeat/train2014_obj36.zip
    wget --no-check-certificate https://nlp1.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/val2014_obj36.zip -P data/mscoco_imgfeat
    unzip data/mscoco_imgfeat/val2014_obj36.zip -d data && rm data/mscoco_imgfeat/val2014_obj36.zip
    ```

4. Before fine-tuning on whole VQA 2.0 training set, verifying the script and model on a small training set (512 images) is recommended. 
The first argument `0` is GPU id. The second argument `vqa_lxr955_tiny` is the name of this experiment.
    ```bash
    bash run/vqa_finetune.bash 0 vqa_lxr955_tiny --tiny
    ```
5. If no bug came out, then the model is ready to be trained on the whole VQA corpus:
    ```bash
    bash run/vqa_finetune.bash 0 vqa_lxr955
    ```
It takes around 8 hours (2 hours per epoch * 4 epochs) to converge. 
The **logs** and **model snapshots** will be saved under folder `snap/vqa/vqa_lxr955`. 
The validation result after training will be around **69.7%** to **70.2%**. 

#### Local Validation
The results on the validation set (our minival set) are printed while training.
The validation result is also saved to `snap/vqa/[experiment-name]/log.log`.
If the log file was accidentally deleted, the validation result in training is also reproducible from the model snapshot:
```bash
bash run/vqa_test.bash 0 vqa_lxr955_results --test minival --load snap/vqa/vqa_lxr955/BEST
```
#### Submitted to VQA test server
1. Download our re-distributed json file containing VQA 2.0 test data.
    ```bash
    wget --no-check-certificate https://nlp1.cs.unc.edu/data/lxmert_data/vqa/test.json -P data/vqa/
    ```
2. Download the faster rcnn features for MS COCO test2015 split (16 GB).
    ```bash
    wget --no-check-certificate https://nlp1.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/test2015_obj36.zip -P data/mscoco_imgfeat
    unzip data/mscoco_imgfeat/test2015_obj36.zip -d data && rm data/mscoco_imgfeat/test2015_obj36.zip
    ```
3. Since VQA submission system requires submitting whole test data, we need to run inference over all test splits 
(i.e., test dev, test standard, test challenge, and test held-out). 
It takes around 10~15 mins to run test inference (448K instances to run).
    ```bash
    bash run/vqa_test.bash 0 vqa_lxr955_results --test test --load snap/vqa/vqa_lxr955/BEST
    ```
 The test results will be saved in `snap/vqa_lxr955_results/test_predict.json`. 
The VQA 2.0 challenge for this year is host on [EvalAI](https://evalai.cloudcv.org/) at [https://evalai.cloudcv.org/web/challenges/challenge-page/163/overview](https://evalai.cloudcv.org/web/challenges/challenge-page/163/overview)
It still allows submission after the challenge ended.
Please check the official website of [VQA Challenge](https://visualqa.org/challenge.html) for detailed information and 
follow the instructions in [EvalAI](https://evalai.cloudcv.org/web/challenges/challenge-page/163/overview) to submit.
In general, after registration, the only thing remaining is to upload the `test_predict.json` file and wait for the result back.

The testing accuracy with exact this code is **72.42%** for test-dev and **72.54%**  for test-standard.
The results with the code base are also publicly shown on the [VQA 2.0 leaderboard](
https://evalai.cloudcv.org/web/challenges/challenge-page/163/leaderboard/498
) with entry `LXMERT github version`.


### GQA

#### Fine-tuning
1. Please make sure the LXMERT pre-trained model is either [downloaded](#pre-trained-models) or [pre-trained](#pre-training).

2. Download the re-distributed json files for GQA balanced version dataset.
The original GQA dataset is available [in the Download section of its website](https://cs.stanford.edu/people/dorarad/gqa/download.html)
and the script to preprocess these datasets is under `data/gqa/process_raw_data_scripts`.
    ```bash
    mkdir -p data/gqa
    wget --no-check-certificate https://https://nlp1.cs.unc.edu/data/lxmert_data/gqa/train.json -P data/gqa/
    wget --no-check-certificate https://https://nlp1.cs.unc.edu/data/lxmert_data/gqa/valid.json -P data/gqa/
    wget --no-check-certificate https://https://nlp1.cs.unc.edu/data/lxmert_data/gqa/testdev.json -P data/gqa/
    ```
3. Download Faster R-CNN features for Visual Genome and GQA testing images (30 GB).
GQA's training and validation data are collected from Visual Genome.
Its testing images come from MS COCO test set (I have verified this with one of GQA authors [Drew A. Hudson](https://www.linkedin.com/in/drew-a-hudson/)).
The image features are
also available on Google Drive and Baidu Drive (see [Alternative Download](#alternative-dataset-and-features-download-links) for details).
    ```bash
    mkdir -p data/vg_gqa_imgfeat
    wget --no-check-certificate https://https://nlp1.cs.unc.edu/data/lxmert_data/vg_gqa_imgfeat/vg_gqa_obj36.zip -P data/vg_gqa_imgfeat
    unzip data/vg_gqa_imgfeat/vg_gqa_obj36.zip -d data && rm data/vg_gqa_imgfeat/vg_gqa_obj36.zip
    wget --no-check-certificate https://https://nlp1.cs.unc.edu/data/lxmert_data/vg_gqa_imgfeat/gqa_testdev_obj36.zip -P data/vg_gqa_imgfeat
    unzip data/vg_gqa_imgfeat/gqa_testdev_obj36.zip -d data && rm data/vg_gqa_imgfeat/gqa_testdev_obj36.zip
    ```

4. Before fine-tuning on whole GQA training+validation set, verifying the script and model on a small training set (512 images) is recommended. 
The first argument `0` is GPU id. The second argument `gqa_lxr955_tiny` is the name of this experiment.
    ```bash
    bash run/gqa_finetune.bash 0 gqa_lxr955_tiny --tiny
    ```

5. If no bug came out, then the model is ready to be trained on the whole GQA corpus (train + validation), and validate on 
the testdev set:
    ```bash
    bash run/gqa_finetune.bash 0 gqa_lxr955
    ```
It takes around 16 hours (4 hours per epoch * 4 epochs) to converge. 
The **logs** and **model snapshots** will be saved under folder `snap/gqa/gqa_lxr955`. 
The validation result after training will be around **59.8%** to **60.1%**. 

#### Local Validation
The results on testdev is printed out while training and saved in `snap/gqa/gqa_lxr955/log.log`.
It could be also re-calculated with command:
```bash
bash run/gqa_test.bash 0 gqa_lxr955_results --load snap/gqa/gqa_lxr955/BEST --test testdev --batchSize 1024
```

> Note: Our local testdev result is usually 0.1% to 0.5% lower than the 
submitted testdev result. 
The reason is that the test server takes an [advanced 
evaluation system](https://cs.stanford.edu/people/dorarad/gqa/evaluate.html) while our local evaluator only 
calculates the exact matching.
Please use [this official evaluator](https://nlp.stanford.edu/data/gqa/eval.zip) (784 MB) if you 
want to have the exact number without submitting.


#### Submitted to GQA test server
1. Download our re-distributed json file containing GQA test data.
    ```bash
    wget --no-check-certificate https://nlp1.cs.unc.edu/data/lxmert_data/gqa/submit.json -P data/gqa/
    ```

2. Since GQA submission system requires submitting the whole test data, 
we need to run inference over all test splits.
It takes around 30~60 mins to run test inference (4.2M instances to run).
    ```bash
    bash run/gqa_test.bash 0 gqa_lxr955_results --load snap/gqa/gqa_lxr955/BEST --test submit --batchSize 1024
    ```

3. After running test script, a json file `submit_predict.json` under `snap/gqa/gqa_lxr955_results` will contain 
all the prediction results and is ready to be submitted.
The GQA challenge 2019 is hosted by [EvalAI](https://evalai.cloudcv.org/) at [https://evalai.cloudcv.org/web/challenges/challenge-page/225/overview](https://evalai.cloudcv.org/web/challenges/challenge-page/225/overview).
After registering the account, uploading the `submit_predict.json` and waiting for the results are the only thing remained.
Please also check [GQA official website](https://cs.stanford.edu/people/dorarad/gqa/) 
in case the test server is changed.

The testing accuracy with exactly this code is **60.00%** for test-dev and **60.33%**  for test-standard.
The results with the code base are also publicly shown on the [GQA leaderboard](
https://evalai.cloudcv.org/web/challenges/challenge-page/225/leaderboard
) with entry `LXMERT github version`.

### NLVR2

#### Fine-tuning

1. Download the NLVR2 data from the official [GitHub repo](https://github.com/lil-lab/nlvr).
    ```bash
    git submodule update --init
    ```


2. Process the NLVR2 data to json files.
    ```bash
    bash -c "cd data/nlvr2/process_raw_data_scripts && python process_dataset.py"
    ```

3. Download the NLVR2 image features for train (21 GB) & valid (1.6 GB) splits. 
The image features are
also available on Google Drive and Baidu Drive (see [Alternative Download](#alternative-dataset-and-features-download-links) for details).
To access to the original images, please follow the instructions on [NLVR2 official Github](https://github.com/lil-lab/nlvr/tree/master/nlvr2).
The images could either be downloaded with the urls or by signing an agreement form for data usage. And the feature could be extracted as described in [feature extraction](#faster-r-cnn-feature-extraction)
    ```bash
    mkdir -p data/nlvr2_imgfeat
    wget --no-check-certificate https://nlp1.cs.unc.edu/data/lxmert_data/nlvr2_imgfeat/train_obj36.zip -P data/nlvr2_imgfeat
    unzip data/nlvr2_imgfeat/train_obj36.zip -d data && rm data/nlvr2_imgfeat/train_obj36.zip
    wget --no-check-certificate https://nlp1.cs.unc.edu/data/lxmert_data/nlvr2_imgfeat/valid_obj36.zip -P data/nlvr2_imgfeat
    unzip data/nlvr2_imgfeat/valid_obj36.zip -d data && rm data/nlvr2_imgfeat/valid_obj36.zip
    ```

4. Before fine-tuning on whole NLVR2 training set, verifying the script and model on a small training set (512 images) is recommended. 
The first argument `0` is GPU id. The second argument `nlvr2_lxr955_tiny` is the name of this experiment.
Do not worry if the result is low (50~55) on this tiny split, 
the whole training data would bring the performance back.
    ```bash
    bash run/nlvr2_finetune.bash 0 nlvr2_lxr955_tiny --tiny
    ```

5. If no bugs are popping up from the previous step, 
it means that the code, the data, and image features are ready.
Please use this command to train on the full training set. 
The result on NLVR2 validation (dev) set would be around **74.0** to **74.5**.
    ```bash
    bash run/nlvr2_finetune.bash 0 nlvr2_lxr955
    ```

#### Inference on Public Test Split
1. Download NLVR2 image features for the public test split (1.6 GB).
    ```bash
    wget --no-check-certificate https://nlp1.cs.unc.edu/data/lxmert_data/nlvr2_imgfeat/test_obj36.zip -P data/nlvr2_imgfeat
    unzip data/nlvr2_imgfeat/test_obj36.zip -d data/nlvr2_imgfeat && rm data/nlvr2_imgfeat/test_obj36.zip
    ```

2. Test on the public test set (corresponding to 'test-P' on [NLVR2 leaderboard](http://lil.nlp.cornell.edu/nlvr/)) with:
    ```bash
    bash run/nlvr2_test.bash 0 nlvr2_lxr955_results --load snap/nlvr2/nlvr2_lxr955/BEST --test test --batchSize 1024
    ```

3. The test accuracy would be shown on the screen after around 5~10 minutes.
It also saves the predictions in the file `test_predict.csv` 
under `snap/nlvr2_lxr955_reuslts`, which is compatible to NLVR2 [official evaluation script](https://github.com/lil-lab/nlvr/tree/master/nlvr2/eval).
The official eval script also calculates consistency ('Cons') besides the accuracy.
We could use this official script to verify the results by running:
    ```bash
    python data/nlvr2/nlvr/nlvr2/eval/metrics.py snap/nlvr2/nlvr2_lxr955_results/test_predict.csv data/nlvr2/nlvr/nlvr2/data/test1.json
    ```

The accuracy of public test ('test-P') set should be almost same to the validation set ('dev'),
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
```bash
# Training with 512 images:
bash run/vqa_finetune.bash 0 --tiny 
# Training with 4096 images:
bash run/vqa_finetune.bash 0 --fast
```

## Pre-training

1. Download our aggregated LXMERT dataset from MS COCO, Visual Genome, VQA, and GQA (around 700MB in total). The joint answer labels are saved in `data/lxmert/all_ans.json`.
    ```bash
    mkdir -p data/lxmert
    wget --no-check-certificate https://nlp1.cs.unc.edu/data/lxmert_data/lxmert/mscoco_train.json -P data/lxmert/
    wget --no-check-certificate https://nlp1.cs.unc.edu/data/lxmert_data/lxmert/mscoco_nominival.json -P data/lxmert/
    wget --no-check-certificate https://nlp1.cs.unc.edu/data/lxmert_data/lxmert/vgnococo.json -P data/lxmert/
    wget --no-check-certificate https://nlp1.cs.unc.edu/data/lxmert_data/lxmert/mscoco_minival.json -P data/lxmert/
    ```

2. [*Skip this if you have run [VQA fine-tuning](#vqa).*] Download the detection features for MS COCO images.
    ```bash
    mkdir -p data/mscoco_imgfeat
    wget --no-check-certificate https://nlp1.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/train2014_obj36.zip -P data/mscoco_imgfeat
    unzip data/mscoco_imgfeat/train2014_obj36.zip -d data/mscoco_imgfeat && rm data/mscoco_imgfeat/train2014_obj36.zip
    wget --no-check-certificate https://nlp1.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/val2014_obj36.zip -P data/mscoco_imgfeat
    unzip data/mscoco_imgfeat/val2014_obj36.zip -d data && rm data/mscoco_imgfeat/val2014_obj36.zip
    ```

3. [*Skip this if you have run [GQA fine-tuning](#gqa).*] Download the detection features for Visual Genome images.
    ```bash
    mkdir -p data/vg_gqa_imgfeat
    wget --no-check-certificate https://nlp1.cs.unc.edu/data/lxmert_data/vg_gqa_imgfeat/vg_gqa_obj36.zip -P data/vg_gqa_imgfeat
    unzip data/vg_gqa_imgfeat/vg_gqa_obj36.zip -d data && rm data/vg_gqa_imgfeat/vg_gqa_obj36.zip
    ```

4. Test on a small split of the MS COCO + Visual Genome datasets:
    ```bash
    bash run/lxmert_pretrain.bash 0,1,2,3 --multiGPU --tiny
    ```

5. Run on the whole [MS COCO](http://cocodataset.org) and [Visual Genome](https://visualgenome.org/) related datasets (i.e., [VQA](https://visualqa.org/), [GQA](https://cs.stanford.edu/people/dorarad/gqa/index.html), [COCO caption](http://cocodataset.org/#captions-2015), [VG Caption](https://visualgenome.org/), [VG QA](https://github.com/yukezhu/visual7w-toolkit)). 
Here, we take a simple single-stage pre-training strategy (20 epochs with all pre-training tasks) rather than the two-stage strategy in our paper (10 epochs without image QA and 10 epochs with image QA).
The pre-training finishes in **8.5 days** on **4 GPUs**.  By the way, I hope that [my experience](experience_in_pretraining.md) in this project would help anyone with limited computational resources.
    ```bash
    bash run/lxmert_pretrain.bash 0,1,2,3 --multiGPU
    ```
    > Multiple GPUs: Argument `0,1,2,3` indicates taking 4 GPUs to pre-train LXMERT. If the server does not have 4 GPUs (I am sorry to hear that), please consider halving the batch-size or using the [NVIDIA/apex](https://github.com/NVIDIA/apex) library to support half-precision computation. 
    The code uses the default data parallelism in PyTorch and thus extensible to less/more GPUs. The python main thread would take charge of the data loading. On 4 GPUs, we do not find that the data loading becomes a bottleneck (around 5% overhead). 
    >
    > GPU Types: We find that either Titan XP, GTX 2080, and Titan V could support this pre-training. However, GTX 1080, with its 11G memory, is a little bit small thus please change the batch_size to 224 (instead of 256).

6. I have **verified these pre-training commands** with 12 epochs. The pre-trained weights from previous process could be downloaded from `https://nlp1.cs.unc.edu/data/github_pretrain/lxmert/EpochXX_LXRT.pth`, XX from `01` to `12`. The results are roughly the same (around 0.3% lower in downstream tasks because of fewer epochs). 

7. Explanation of arguments in the pre-training script `run/lxmert_pretrain.bash`:
    ```bash
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
        --batchSize 256 --optim bert --lr 1e-4 --epochs 20 \
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
- All code is in folder `src`. The basics in `lxrt`.
The python files related to pre-training and fine-tuning are saved in `src/pretrain` and `src/tasks` respectively.
- I kept folders containing image features (e.g., mscoco_imgfeat) separated from vision-and-language dataset (e.g., vqa, lxmert) because
multiple vision-and-language datasets would share common images.
- We use the name `lxmert` for our framework and use the name `lxrt`
(Language, Cross-Modality, and object-Relationship Transformers) to refer to our our models.
- To be consistent with the name `lxrt` (Language, Cross-Modality, and object-Relationship Transformers), 
we use `lxrXXX` to denote the number of layers.
E.g., `lxr955` (used in current pre-trained model) indicates 
a model with 9 Language layers, 5 cross-modality layers, and 5 object-Relationship layers. 
If we consider a single-modality layer as a half of cross-modality layer, 
the total number of layers is `(9 + 5) / 2 + 5 = 12`, which is the same as `BERT_BASE`.
- We share the weight between the two cross-modality attention sub-layers. Please check the [`visual_attention` variable](blob/master/src/lxrt/modeling.py#L521), which is used to compute both `lang->visn` attention and `visn->lang` attention. (I am sorry that the name `visual_attention` is misleading because I deleted the `lang_attention` there.) Sharing weights is mostly used for saving computational resources and it also (intuitively) helps forcing the features from visn/lang into a joint subspace.
- The box coordinates are not normalized from [0, 1] to [-1, 1], which looks like a typo but actually not ;). Normalizing the coordinate would not affect the output of box encoder (mathematically and almost numerically). ~~(Hint: consider the LayerNorm in positional encoding)~~


## Faster R-CNN Feature Extraction


We use the Faster R-CNN feature extractor demonstrated in ["Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering", CVPR 2018](https://arxiv.org/abs/1707.07998)
and its released code at [Bottom-Up-Attention github repo](https://github.com/peteanderson80/bottom-up-attention).
It was trained on [Visual Genome](https://visualgenome.org/) dataset and implemented based on a specific [Caffe](https://caffe.berkeleyvision.org/) version.


To extract features with this Caffe Faster R-CNN, we publicly release a docker image `airsplay/bottom-up-attention` on docker hub that takes care of all the dependencies and library installation . Instructions and examples are demonstrated below. You could also follow the installation instructions in the bottom-up attention github to setup the tool: [https://github.com/peteanderson80/bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention). 

The BUTD feature extractor is widely used in many other projects. If you want to reproduce the results from their paper, feel free to use our docker as a tool.


### Feature Extraction with Docker
[Docker](https://www.docker.com/) is a easy-to-use virtualization tool which allows you to plug and play without installing libraries.

The built docker file for bottom-up-attention is released on [docker hub](https://hub.docker.com/r/airsplay/bottom-up-attention) and could be downloaded with command: 
```bash
sudo docker pull airsplay/bottom-up-attention
```
> The `Dockerfile` could be downloaed [here](https://drive.google.com/file/d/1KJjwQtqisXvinWm8OORk-_3XYLBHYCIK/view?usp=sharing), which allows using other CUDA versions.

After pulling the docker, you could test running the docker container with command:
```bash
docker run --gpus all --rm -it airsplay/bottom-up-attention bash
``` 


If errors about `--gpus all` popped up, please read the next section.

#### Docker GPU Access
Note that the purpose of the argument `--gpus all` is to expose GPU devices to the docker container, and it requires Docker >= 19.03 along with `nvidia-container-toolkit`:
1. [Docker CE 19.03](https://docs.docker.com/install/linux/docker-ce/ubuntu/)
2. [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-docker)

For running Docker with an older version, either update it to 19.03 or use the flag `--runtime=nvidia` instead of `--gpus all`.

#### An Example: Feature Extraction for NLVR2 
We demonstrate how to extract Faster R-CNN features of NLVR2 images.

1. Please first follow the instructions on the [NLVR2 official repo](https://github.com/lil-lab/nlvr/tree/master/nlvr2) to get the images.

2. Download the pre-trained Faster R-CNN model. Instead of using the default pre-trained model (trained with 10 to 100 boxes), we use the ['alternative pretrained model'](https://github.com/peteanderson80/bottom-up-attention#demo) which was trained with 36 boxes. 
    ```bash
    wget --no-check-certificate 'https://www.dropbox.com/s/nu6jwhc88ujbw1v/resnet101_faster_rcnn_final_iter_320000.caffemodel?dl=1' -O data/nlvr2_imgfeat/resnet101_faster_rcnn_final_iter_320000.caffemodel
    ```

3. Run docker container with command:
    ```bash
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


4. Extract the features **inside the docker container**. The extraction script is copied from [butd/tools/generate_tsv.py](https://github.com/peteanderson80/bottom-up-attention/blob/master/tools/generate_tsv.py) and modified by [Jie Lei](http://www.cs.unc.edu/~jielei/) and me.
    ```bash
    cd /workspace/features
    CUDA_VISIBLE_DEVICES=0 python extract_nlvr2_image.py --split train 
    CUDA_VISIBLE_DEVICES=0 python extract_nlvr2_image.py --split valid
    CUDA_VISIBLE_DEVICES=0 python extract_nlvr2_image.py --split test
    ```

5. It would takes around 5 to 6 hours for the training split and 1 to 2 hours for the valid and test splits. Since it is slow, I recommend to run them parallelly if there are multiple GPUs. It could be achieved by changing the `gpu_id` in `CUDA_VISIBLE_DEVICES=$gpu_id`.

The features will be saved in `train.tsv`, `valid.tsv`, and `test.tsv` under the directory `data/nlvr2_imgfeat`, outside the docker container. I have verified the extracted image features are the same to the ones I provided in [NLVR2 fine-tuning](#nlvr2).

#### Yet Another Example: Feature Extraction for MS COCO Images
1. Download the MS COCO train2014, val2014, and test2015 images from [MS COCO official website](http://cocodataset.org/#download).

2. Download the pre-trained Faster R-CNN model. 
    ```bash
    mkdir -p data/mscoco_imgfeat
    wget --no-check-certificate 'https://www.dropbox.com/s/nu6jwhc88ujbw1v/resnet101_faster_rcnn_final_iter_320000.caffemodel?dl=1' -O data/mscoco_imgfeat/resnet101_faster_rcnn_final_iter_320000.caffemodel
    ```

3. Run the docker container with the command:
    ```bash
    docker run --gpus all -v /path/to/mscoco/images:/workspace/images:ro -v $(pwd)/data/mscoco_imgfeat:/workspace/features --rm -it airsplay/bottom-up-attention bash
    ```
    > Note: Option `-v` mounts the folders outside container to the paths inside the container.
    > 
    > Note1: Please use the **absolute path** to the MS COCO images folder `images`. The `images` folder containing the `train2014`, `val2014`, and `test2015` sub-folders. (It's the standard way to save MS COCO images.)

4. Extract the features **inside the docker container**.
    ```bash
    cd /workspace/features
    CUDA_VISIBLE_DEVICES=0 python extract_coco_image.py --split train 
    CUDA_VISIBLE_DEVICES=0 python extract_coco_image.py --split valid
    CUDA_VISIBLE_DEVICES=0 python extract_coco_image.py --split test
    ```
 
5. Exit from the docker container (by executing `exit` command in bash). The extracted features would be saved under folder `data/mscoco_imgfeat`. 


## Reference
If you find this project helps, please cite our paper :)

```bibtex
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
[Bottom-Up-Attention Github Repo](https://github.com/peteanderson80/bottom-up-attention).  We thank [Hengyuan Hu](https://www.linkedin.com/in/hengyuan-hu-8963b313b) for his [PyTorch VQA](https://github.com/hengyuan-hu/bottom-up-attention-vqa) implementation, our VQA implementation borrows its pre-processed answers.
We thank [hugginface](https://github.com/huggingface) for releasing the excellent PyTorch code 
[PyTorch Transformers](https://github.com/huggingface/pytorch-transformers).  

We thank [Drew A. Hudson](https://www.linkedin.com/in/drew-a-hudson/) to answer all our questions about GQA specification.
We thank [Alane Suhr](http://alanesuhr.com/) for helping test LXMERT on NLVR2 unreleased test split and provide [a detailed analysis](http://lil.nlp.cornell.edu/nlvr/NLVR2BiasAnalysis.html).

We thank all the authors and annotators of vision-and-language datasets 
(i.e., 
[MS COCO](http://cocodataset.org/#home), 
[Visual Genome](https://visualgenome.org/),
[VQA](https://visualqa.org/),
[GQA](https://cs.stanford.edu/people/dorarad/gqa/),
[NLVR2](http://lil.nlp.cornell.edu/nlvr/)
), 
which allows us to develop a pre-trained model for vision-and-language tasks.

We thank [Jie Lei](http://www.cs.unc.edu/~jielei/) and [Licheng Yu](http://www.cs.unc.edu/~licheng/) for their helpful discussions. I also want to thank [Shaoqing Ren](https://www.shaoqingren.com/) to teach me vision knowledge when I was in MSRA.  We also thank you to help look into our code. Please kindly contact us if you find any issue. Comments are always welcome.

LXRThanks.
