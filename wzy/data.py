import os
import os.path as osp
import shutil
import tqdm

data_root="/mnt/100_data6/xiaomi/data/mvtec_ad"


dir_lis=list(filter(lambda x: osp.isdir(osp.join(data_root,x)),os.listdir(data_root)))

for dir in tqdm.tqdm(dir_lis):
    for file in os.listdir(osp.join(data_root,dir,"test/good")):
        shutil.copy2(osp.join(data_root,dir,"test/good",file),osp.join(data_root,"../vqkd_data/test",f"{dir}_{file}"))