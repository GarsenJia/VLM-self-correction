# VLM-self-correction

## TODO:  
how exactly to fine-tune? - Sola​  
experiment design/do we have enough compute -Shengyi​  
what makes a good CoT prompt - Yining​  

llava: need to adjust prompting to llava. it sometimes will directly give out the answer, and the structure and content of the hint needs to be standardized. - Ruchen

Maybe we can start working on the final report. Here are some pre-results might be useful:
before finetune:
eval on the test + train for first inference
Total Entries: 3700
Entries with True Percentage > 0: 564
Average True Percentage (non-zero only): 9.99
max percentage: 63.33333333333333
min percentage: 3.3333333333333335

eval on the test for first inference
Total Entries: 1000
Entries with True Percentage > 0: 182
Average True Percentage (non-zero only): 10.44
max percentage: 43.333333333333336
min percentage: 3.3333333333333335

eval on the train for first inference
Total Entries: 2700
Entries with True Percentage > 0: 378
Average True Percentage (non-zero only): 9.76
max percentage: 63.33333333333333
min percentage: 3.3333333333333335

after finetune: finetune time per epoch is about 6 mins on 1 A100 GPU
Total Entries: 1000
Entries with True Percentage > 0: 134
Average True Percentage (non-zero only): 7.51
max percentage: 36.666666666666664
min percentage: 3.3333333333333335

next, we need hints for these 1000 examples and see if the model can perform a bit better

Let's see the data of the larger model:
Total Entries: 1000
Entries with True Percentage > 0: 187
Average True Percentage (non-zero only): 19.20
max percentage: 73.33333333333333
min percentage: 3.3333333333333335
It seems that it improves a bit but not too much. 

Then let's finetune the large model : time is about 13 min per epoch on 1 A100GPU
Total Entries: 1000
Entries with True Percentage > 0: 107
Average True Percentage (non-zero only): 9.31
max percentage: 53.333333333333336
min percentage: 3.3333333333333335

hmm the performance also decreases. This is actually expected!

now, let's do the experiment of second inference directly on the given **hints data**.
inference time: 10 min/500 examples
MODEL_PATH = "Florence-2-CoTVMCQA_model_6_epochs"
PROCESSOR_PATH = "Florence-2-CoTVMCQA_processor_6_epochs"
Total Entries: 500
Non-Zero Percentages: 371
Average True Percentage (Non-Zero Only): 15.91
Max True Percentage: 60.00
Min True Percentage: 3.33
cool, we seem to have sth work!

Next, how about do this for the base model?
inference time: 24 min/500 examples


