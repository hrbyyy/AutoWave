# AutoWave
1.A2,ECG, kpi directories represent experiments on the three datasets.

2.For each dataset, run data_pre first to perform data preparation, then run main_dataset.py to train AutoWave and make inference.Finally, run post_process to calculate relevant metrics followed by sta.py to compute the average performance of repeated 5 experiments.

3.A2, ECG and kpi dataset can be downloaded from https://yahooresearch.tumblr.com/post/114590420346/a-benchmarkdataset-for-time-series-anomaly,
 http://iops.ai/competition detail/?competition id=5&flag=1 respectively. For ECG dataset, we perform preprocessing based on data from https://github.com/Vniex/BeatGAN.
