# Running Script Arguments

```
Data Splits: 
    --train [str,str,...]: use the splits (separated by comma) in training.
    --valid [str,str,...]: use the splits (separated by comma) in validation.
    --test [str,str,...]: use the splits (separated by comma) in testing.
Model Architecture:
    --llayers [int]: number of layers in language encoder.
    --xlayers [int]: number of layers in cross-modality encoder.
    --rlayers [int]: number of layers in object relationship encoder.
Load Weights:
    --load [str='path/to/saved_model']: load fine-tuned model path/to/saved_model.pth.
    --loadLXMERT [str='path/to/saved_model']: load pre-trained model without answer heads from path/to/saved_model_LXRT.pth.
    --loadLXMERTQA [str='path/to/saved_model']: load pre-trained model with answer head path/to/saved_model_LXRT.pth.
    --fromScratch: If none of the above loading parameters are set, the default mode would 
      load the pre-trained BERT weights.
      As we promised to EMNLP reviewers, the language encoder would be re-initialized with this one-line argument to test the performance without BERT weights.
Training Hyper Parameters:
    --batchSize [int]: batch size.
    --optim [str]: optimizers.
    --lr [float]: peak learning rate.
    --epochs [int]: training epochs.
Debugging:
    --tiny: Load 512 images for each data split. (Note: number of images might be changed due to dataset specification)
    --fast: Load 5000 images for each data split. (Note: number of images might be changed due to dataset specification)
```

# Pre-training-Specific Arguments
```
Pre-training Tasks:
    --taskMaskLM: use the masked language model task.
    --taskObjPredict: use the masked object prediction task.
    --taskMatched: use the cross-modality matched task.
    --taskQA: use the image QA task.
Visual Pre-training Losses (Tasks):
    --visualLosses [str,str,...]: The sub-tasks in pre-training visual modality. Each one is from 'obj,attr,feat'. 
      obj: detected-object-label classification. 
      attr: detected-object-attribute classification. 
      feat: RoI-feature regression.
Mask Rate in Pre-training:
    --wordMaskRate [float]: The prob of masking a word.
    --objMaskRate [float]: The prob of masking an object.
Initialization:
    --fromScratch: The default mode would load the pre-trained BERT weights into the model. 
      As we promised to EMNLP reviewers, this option would re-initialize the language encoder.
```


