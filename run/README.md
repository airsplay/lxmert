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
    --load path/to/saved_model: load fine-tuned model path/to/saved_model.pth.
    --loadLXMERT path/to/saved_model: load pre-trained model without answer heads from path/to/saved_model_LXRT.pth.
    --loadLXMERTQA path/to/saved_model: load pre-trained model with answer head path/to/saved_model_LXRT.pth.
Training Hyper Parameters
    --batchSize [int]: batch size.
    --optim [str]: optimizers.
    --lr [float]: peak learning rate.
    --epochs [int]: training epochs.
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
    --fromScratch: The default mode would train with loaded BERT weights. 
      As we promised to the EMNLP reviewer, the model would re-initialized with this one-line argument.
```


