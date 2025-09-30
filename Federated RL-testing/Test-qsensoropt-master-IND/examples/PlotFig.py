import pandas as pd
import os

folder_path = ("indNNi70/data")
output_file = "indNNi70/merged_output.csv"


Mse_lists =[]
nnMse_lists =[]
for i in range(0,10):
    file_paths = folder_path+str(i)
    file_path = file_paths + ("/nv_center_time_invT2_0.0104_lr_0.001_batchsize_64_num_steps"
                              "_10000_max_resources_10000.00_ll_True_cl_True_eval.csv")
    df = pd.read_csv(file_path)
    res = df['Resources']
    Mse_lists.append(df['MSE'])


Mse_mean =  sum(Mse_lists) / len(Mse_lists)


merged_df = pd.DataFrame({'res': res,'mean_mse': Mse_mean})
merged_df.to_csv(output_file, index=False)
print(16*127,Mse_mean,f"Merged data saved to {output_file}")
