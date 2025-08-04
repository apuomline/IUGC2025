# import os
# import pandas as pd

# # 配置
# img_dir = r'pse_training_inputs\pse_825_se_100_le_220_unimatched_37\val_imgs_12'         # 图像文件夹路径
# input_csv = r'inputs\label.csv'           # 包含Filename列的csv
# output_csv = 'pse_825_100_le220_val_12.csv'       # 输出的新csv

# # 获取文件夹下所有图像文件名（不含路径）
# img_filenames = set(os.listdir(img_dir))

# # 读取csv
# df = pd.read_csv(input_csv)

# # 只保留Filename在文件夹中的行
# matched_df = df[df['Filename'].isin(img_filenames)]

# # 保存到新csv
# matched_df.to_csv(output_csv, index=False)
# print(f"Saved {len(matched_df)} matched rows to {output_csv}")



import os
import pandas as pd

# 配置
img_dir = r'pse_training_inputs\pse_825_se_100_le_220_unimatched_37\val_imgs_12'         # 图像文件夹
csv_file = r'pse_training_inputs\pse_825_se_100_le_220_unimatched_37\val_68.csv'         # 包含Filename列的csv

# 1. 读取文件夹下所有图像文件名
img_filenames = set(os.listdir(img_dir))

# 2. 读取csv
df = pd.read_csv(csv_file)

# 3. 找到需要删除的行
to_delete = df['Filename'].isin(img_filenames)

# 4. 删除这些行
filtered_df = df[~to_delete]

# 5. 覆盖原csv文件
filtered_df.to_csv(csv_file, index=False)
print(f"Overwrote {csv_file}, {to_delete.sum()} rows removed.")