# Running Arguments

## Splits
```
--train X: using split X in training 
--valid X: using split X in validation
--test X: using split X in testing
```

## Model
```
--llayers X: Using X (a number) layers in language encoder.
--xlayers X: Using X (a number) layers in cross-modality encoder.
--rlayers X: Using X (a number) layers in object relationship encoder.
```

## Load
```
--load X: load fine-tuned model X.
--loadLXMERT X: load pre-trained model without answer heads.
--loadLXMERTQA X: load pre-trained model with answer heads.
```

## Hyper parameters
```
--batchSize X: Batch size
--optim X: Optimizers.
--lr X: peak learning rate
--epochs X: train for X epochs
```

#Pre-training Arguments
## Pre-training tasks:
```
--taskMaskLM: use the masked language model task.
--taskObjPredict: use the masked object prediction task.
--taskMatched: use the cross-modality matched task.
--taskQA: use the image QA task.
```


