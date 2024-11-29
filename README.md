# VLM-self-correction


## WE HAVE ALMOST FINISHED THE FINAL PROJECT!!! BELOW ARE SOME DATA FOR ANALYSIS: SEE THE results_summary.txt for all info

==================================== applying new metric
**NOW BEGINS THE REAL DATA!!!**
results for no revision added
1. base
=========================================
Total Entries: 1000
Non-Zero Percentages: 316
Average True Percentage (Non-Zero Only): 18.42
Max True Percentage: 76.67
Min True Percentage: 3.33
2. large
=========================================
Total Entries: 1000
Non-Zero Percentages: 366
Average True Percentage (Non-Zero Only): 21.50
Max True Percentage: 76.67
Min True Percentage: 3.33

results for only prompting:
1. before finetune:
==========================================
large
Total Entries: 1000
Non-Zero Percentages: 259
Average True Percentage (Non-Zero Only): 12.20
Max True Percentage: 63.33
Min True Percentage: 3.33
===========================================
base
Total Entries: 1000
Non-Zero Percentages: 205
Average True Percentage (Non-Zero Only): 8.39
Max True Percentage: 40.00
Min True Percentage: 3.33
============================================
results for only finetune:
large:
Total Entries: 1000
Non-Zero Percentages: 704
Average True Percentage (Non-Zero Only): 33.93
Max True Percentage: 100.00
Min True Percentage: 3.33
base:
Total Entries: 1000
Non-Zero Percentages: 742
Average True Percentage (Non-Zero Only): 16.85
Max True Percentage: 66.67
Min True Percentage: 3.33

results for prompting + finetune:
1. base:
============================================
Total Entries: 1000
Non-Zero Percentages: 661
Average True Percentage (Non-Zero Only): 9.60
Max True Percentage: 43.33
Min True Percentage: 3.33
2. large:
=============================================
Total Entries: 1000
Non-Zero Percentages: 668
Average True Percentage (Non-Zero Only): 26.03
Max True Percentage: 100.00
Min True Percentage: 3.33

results for hints + finetune:

==========================================
large:
Total Entries: 1000
Non-Zero Percentages: 712
Average True Percentage (Non-Zero Only): 28.27
Max True Percentage: 96.67
Min True Percentage: 3.33

base:
Total Entries: 1000
Non-Zero Percentages: 756
Average True Percentage (Non-Zero Only): 16.28
Max True Percentage: 70.00
Min True Percentage: 3.33

===========================================
results for textonly:
base:
Total Entries: 1000
Non-Zero Percentages: 697
Average True Percentage (Non-Zero Only): 9.18
Max True Percentage: 36.67
Min True Percentage: 3.33

large
Total Entries: 1000
Non-Zero Percentages: 584
Average True Percentage (Non-Zero Only): 10.55
Max True Percentage: 83.33
Min True Percentage: 3.33

==============================================
results for hints + not finetuned (raw model with hints)
base:
Total Entries: 1000
Non-Zero Percentages: 298
Average True Percentage (Non-Zero Only): 14.24
Max True Percentage: 66.67
Min True Percentage: 3.33

large:
Total Entries: 1000
Non-Zero Percentages: 354
Average True Percentage (Non-Zero Only): 17.90
Max True Percentage: 80.00
Min True Percentage: 3.33

FINISHED!


## TODO:  
how exactly to fine-tune? - Sola​  
experiment design/do we have enough compute -Shengyi​  
what makes a good CoT prompt - Yining​  

llava: need to adjust prompting to llava. it sometimes will directly give out the answer, and the structure and content of the hint needs to be standardized. - Ruchen
