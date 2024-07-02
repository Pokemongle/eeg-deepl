import torch
from tqdm.notebook import tqdm
import os
import scipy.io
import gc
from load_data import MyData  # self-made


def generate_eegmap(dataset, matrix_index, exper_dir, condi_dir, device):
    for person in tqdm(range(len(dataset))):  # 12种情况中的1种，其中的所有被试
        # filepath是最子文件夹中每个.mat文件的名字
        # path是包含当前情况的.mat文件的子文件夹
        filename = os.path.join(dataset.path, dataset.file_path[person])  # eg. conditionA\hc\hc1.set.mat
        data = scipy.io.loadmat(filename)  # 读取该被试的数据为字典
        data = data['datas']  # 键值对读取59*2400*[paras]的数据矩阵
        # 将数据移动到GPU上
        data = torch.from_numpy(data).to(device)

        # 在GPU上处理数据
        # 创建被试全0map
        data_map_person = []
        print(f"----现在处理：{dataset.file_path[person]}，共{data.shape[2]}段----")
        # 遍历该被试的所有段
        for para in tqdm(range(data.shape[2])):
            # 创建段全0map
            data_map_para = []
            for point in range(data.shape[1]):  # 遍历该段所有数据点
                # 创建单个数据点全0map
                data_map_point = torch.zeros((10, 11), device=device)
                # 遍历59电极数据并赋值给全0map
                for channel in range(data.shape[0]):
                    data_map_point[matrix_index[0][channel]][matrix_index[1][channel]] = data[channel][point][para]
                # 保存单个数据点map到段map列表
                data_map_para.append(data_map_point)
            # 保存 段map 到 被试map
            data_map_person.append(torch.stack(data_map_para))
            print(len(data_map_person))
            # 清理内存
            del data_map_para
            gc.collect()
            torch.cuda.empty_cache()
        # 保存 被试map 到文件中
        save_path = f"../eegmap_direct/{exper_dir}/{condi_dir}/{dataset.file_path[person]}.pt"
        torch.save(torch.stack(data_map_person), save_path)


def generate_eegmap_chunks(root_dir, exper_dir, condi_dir):
    dataset = MyData(root_dir, exper_dir, condi_dir)
    data_all = []
    for person in tqdm(range(len(dataset))):
        filename = os.path.join(dataset.path, dataset.file_path[person])
        data_map = torch.load(filename)
        print(filename)
        print(data_map.size())

        for i in range(data_map.size(0)):  # 每个被试切割成2400大小的块保存到data_all中
            data_all.append(data_map[i])

    savepath = f"../data/eegmap_chunks/{exper_dir}/{condi_dir}/{exper_dir}_{condi_dir}.pt"
    torch.save(torch.stack(data_all), savepath)
    print(len(data_all))
    del data_all
    del dataset
    del data_map
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    print("generate eegmap")