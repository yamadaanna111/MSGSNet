import argparse
import os
import h5py
from time import perf_counter as pc
from datetime import timedelta
from seq2graph import create_graph
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--ksize", type=int, required=True, help="K-size")
    args = parser.parse_args()
    k = args.ksize
    return k


def load_data(file_path):
    sequences = []
    labels = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 3:
                print(f"Warning: Skipping invalid line: {line.strip()} (Expected at least 3 columns)")
                continue
            sequence = parts[1]
            label = parts[2]
            sequences.append(sequence)
            labels.append(int(label))

    print("Dataset:", file_path)
    print("Number of sequences:", len(sequences))
    print("Number of classes:", len(set(labels)))
    return sequences, labels


def create_graphs(sequences, labels, k, dataset_type, dataset_name):
    graphs = []
    print(f"Building graphs for {dataset_name} {dataset_type} dataset...")
    start = pc()

    for s, l in tqdm(zip(sequences, labels), total=len(sequences)):
        seq = s.decode('utf8') if isinstance(s, bytes) else s
        g = create_graph(seq, k)
        g.y = torch.tensor(l)
        graphs.append(g)

    end = pc()
    total_time = timedelta(seconds=end - start)
    print(f"Completed in: {str(total_time)}")

    # 创建按数据集名称组织的目录结构
    base_dir = "../experiments/"
    dataset_dir = os.path.join(base_dir, dataset_name, f"{k}_mer")
    os.makedirs(dataset_dir, exist_ok=True)

    # 保存图数据
    graph_path = os.path.join(dataset_dir, f"{dataset_type}_dataset_{k}.pickle")
    with open(graph_path, "wb") as f:
        pickle.dump(graphs, f)

    # 保存元信息
    info_path = os.path.join(dataset_dir, f"{dataset_type}_dataset_{k}.txt")
    with open(info_path, 'w') as f:
        f.write(f"{dataset_name} {dataset_type} Dataset\n")
        f.write(f"Number of sequences: {len(graphs)}\n")
        f.write(f"Total time: {str(total_time)}\n")


def get_dataset_list():
    """从FILESLIST.txt文件中读取数据集名称列表"""
    fileslist_path = "/mnt/share/wzy/GCBLANE/Datasets/165 ChIP-seq/FILESLIST.txt"

    if not os.path.exists(fileslist_path):
        print(f"Error: FILESLIST.txt not found at {fileslist_path}")
        return []

    dataset_list = []
    with open(fileslist_path, 'r') as file:
        for line in file:
            dataset_name = line.strip()
            # 跳过空行和注释行
            if dataset_name and not dataset_name.startswith('#'):
                dataset_list.append(dataset_name)

    print(f"Loaded {len(dataset_list)} datasets from FILESLIST.txt")
    return dataset_list


def process_dataset(k, dataset_name):
    """处理单个数据集的训练和测试数据"""
    base_path = "/mnt/share/wzy/GCBLANE/Datasets/165 ChIP-seq/"

    # 构建完整路径
    train_file = os.path.join(base_path, dataset_name, "train.data")
    test_file = os.path.join(base_path, dataset_name, "test.data")

    try:
        # 处理训练数据
        if os.path.exists(train_file):
            train_sequences, train_labels = load_data(train_file)
            create_graphs(train_sequences, train_labels, k, "train", dataset_name)
        else:
            print(f"Warning: Train file not found for {dataset_name}")

        # 处理测试数据
        if os.path.exists(test_file):
            test_sequences, test_labels = load_data(test_file)
            create_graphs(test_sequences, test_labels, k, "test", dataset_name)
        else:
            print(f"Warning: Test file not found for {dataset_name}")

    except Exception as e:
        print(f"Error processing {dataset_name}: {str(e)}")


def main():
    k = parse_arguments()
    dataset_list = get_dataset_list()

    if not dataset_list:
        print("No datasets found. Exiting.")
        return

    print(f"Starting graph generation for {len(dataset_list)} datasets with k={k}")

    for dataset_name in dataset_list:
        print(f"\n{'=' * 50}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'=' * 50}")

        process_dataset(k, dataset_name)

    print("\nAll datasets processed successfully!")


if __name__ == "__main__":
    main()