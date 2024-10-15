import os
from tqdm import tqdm
path = '/mnt/wbz/zhoujingkai.zjk/data/image_dataset/download_dance/train'
cnt = 0
for clipfile in tqdm(os.listdir(path)):
    if clipfile.endswith('.pkl'):
        continue
    denseposepath = os.path.join(path, clipfile, 'densepose')
    imagepath = os.path.join(path, clipfile, 'image')

    files_dir1 = set([file.split('_')[-1] for file in os.listdir(denseposepath)])
    files_dir2 = set([file.split('_')[-1] for file in os.listdir(imagepath)])
    # 比较文件列表并找出差异
    only_in_dir1 = files_dir1 - files_dir2
    only_in_dir2 = files_dir2 - files_dir1
    if only_in_dir1 or only_in_dir2:
        print(os.path.join(path, clipfile))

    # if  os.path.exists(imagepath):
    #     print(imagepath)
    #     # print(os.listdir(os.path.join(path, clipfile, 'densepose','img')))
    #     cnt+=1
# print(cnt)

