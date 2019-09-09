# Experience in Pre-training
Since I finish this project with quite limited computational resources, I would like to share some experiences.  If you are also in a small group and plan to pre-train back-bone models for fun, hope it would help.

## Workflow
1. Design a model and its pre-training strategies.
2. Test whether the code is correct or not by over-fitting a super small split (5000 images, typically) of aggregated data.
3. Pre-train it on **all aggregated pre-training data** for around 3 to 4 epochs. (At least make sure that all the images are included!)
4. Test the pre-training performance on a **small split** of fine-tuning tasks.
5. If the accuracy (i.e., results) of the fine-tuning tasks keep growing, it indicates that the pre-training is effective!
6. Compare the **full fine-tuning-data** results when 3-4 epochs' pre-training finishes and select the best pre-training strategies. 
7. Train on **full aggregated data** and have a good one-week sleep ;).


## Tips
- **Do not** verify pre-training strategies (pre-training tasks, pre-training model) on a **small split** of the data. The behavior of pre-training on a small split is significantly different from the full pre-training dataset. 
- Do not over-tune the pre-training hyperparameters. Keep in mind that a good idea will overshadow all these cherry-pick hyper-parameters. Anyway, you would not have enough GPUs to do that.
- Add a component at each time; Have a plan for it.
- Pipeline everything.
- You could rest but GPUs never get rest; GPUs are sometimes broken but you never give up.
