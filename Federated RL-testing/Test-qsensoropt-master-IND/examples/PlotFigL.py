import pandas as pd
import os

folder_path = ("InDomg20/range")
output_file = "InDomg20/merged_outputLoss.csv"


Mse_lists =[]
nnMse_lists =[]
for i in range(0,10):
    file_paths = folder_path+str(i)+"/trained_models"
    file_path = file_paths + ("/nv_center_time_invT2_0.0104_lr_0.001_batchsize_1024_num_"
                              "steps_10000_max_resources_10000.00_ll_True_cl_True_history.csv")
    df = pd.read_csv(file_path)

    Mse_lists.append(df['Loss'])


Mse_mean =  sum(Mse_lists) / len(Mse_lists)


merged_df = pd.DataFrame({'mean_loss': Mse_mean})
merged_df.to_csv(output_file, index=False)
print(16*127,Mse_mean,f"Merged data saved to {output_file}")
