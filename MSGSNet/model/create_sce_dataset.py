import os
import pickle
import numpy as np
from tqdm import tqdm

# ========== Codon Encoding Dictionary ==========
coden_dict1 = {
    'GCT': 0, 'GCC': 0, 'GCA': 0, 'GCG': 0,  # alanine<A>
    'TGT': 1, 'TGC': 1,  # cysteine<C>
    'GAT': 2, 'GAC': 2,  # aspartic acid<D>
    'GAA': 3, 'GAG': 3,  # glutamic acid<E>
    'TTT': 4, 'TTC': 4,  # phenylalanine<F>
    'GGT': 5, 'GGC': 5, 'GGA': 5, 'GGG': 5,  # glycine<G>
    'CAT': 6, 'CAC': 6,  # histidine<H>
    'ATT': 7, 'ATC': 7, 'ATA': 7,  # isoleucine<I>
    'AAA': 8, 'AAG': 8,  # lysine<K>
    'TTA': 9, 'TTG': 9, 'CTT': 9, 'CTC': 9, 'CTA': 9, 'CTG': 9,  # leucine<L>
    'ATG': 10,  # methionine<M>
    'AAT': 11, 'AAC': 11,  # asparagine<N>
    'CCT': 12, 'CCC': 12, 'CCA': 12, 'CCG': 12,  # proline<P>
    'CAA': 13, 'CAG': 13,  # glutamine<Q>
    'CGT': 14, 'CGC': 14, 'CGA': 14, 'CGG': 14, 'AGA': 14, 'AGG': 14,  # arginine<R>
    'TCT': 15, 'TCC': 15, 'TCA': 15, 'TCG': 15, 'AGT': 15, 'AGC': 15,  # serine<S>
    'ACT': 16, 'ACC': 16, 'ACA': 16, 'ACG': 16,  # threonine<T>
    'GTT': 17, 'GTC': 17, 'GTA': 17, 'GTG': 17,  # valine<V>
    'TGG': 18,  # tryptophan<W>
    'TAT': 19, 'TAC': 19,  # tyrosine<Y>
    'TAA': 20, 'TAG': 20, 'TGA': 20,  # STOP code
}

# ========== SCE 编码 + Padding ==========
def coden1_pad(seq, max_len=None):
    """Convert DNA sequence to SCE one-hot encoding with padding/truncation"""
    seq = seq.upper().replace('N', '')  # remove ambiguous bases
    if max_len is None:
        max_len = len(seq) - 2  # 自动计算codon数
    vectors = np.zeros((max_len, 21))
    for i in range(min(len(seq) - 2, max_len)):
        codon = seq[i:i+3]
        if codon in coden_dict1:
            vectors[i][coden_dict1[codon]] = 1
    return vectors

# ========== 路径配置 ==========
base_data_dir = "/mnt/share/wzy/GCBLANE/Datasets/165 ChIP-seq"
base_save_dir = "/mnt/share/wzy/GCBLANE/experiments"
# base_data_dir = "/media/lichangyong/mnt/share2/wzy/GCBLANE/Datasets/165 ChIP-seq"
# base_save_dir = "/media/lichangyong/mnt/share2/wzy/GCBLANE/experiments"
os.makedirs(base_save_dir, exist_ok=True)

# ========== 读取数据集列表 ==========
fileslist_path = os.path.join(base_data_dir, "FILESLIST.txt")
with open(fileslist_path, "r") as f:
    dataset_names = [line.strip() for line in f if line.strip()]

print(f"共找到 {len(dataset_names)} 个数据集")

# ========== 遍历每个数据集 ==========
for dataset_name in dataset_names:
    print(f"\n===== 处理数据集: {dataset_name} =====")
    src_dir = os.path.join(base_data_dir, dataset_name)
    dst_dir = os.path.join(base_save_dir, dataset_name, "SCE")
    os.makedirs(dst_dir, exist_ok=True)

    for split in ["train", "test"]:
        src_path = os.path.join(src_dir, f"{split}.data")
        if not os.path.exists(src_path):
            print(f"[Skip] {src_path} 不存在")
            continue

        print(f"加载 {src_path}")
        sequences, labels = [], []
        with open(src_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    _id, seq, label = line.split()
                    sequences.append(seq)
                    labels.append(int(label))
                except ValueError:
                    print(f"[Warning] 跳过无法解析的行: {line[:60]}")
                    continue

        print(f"样本数: {len(sequences)}")
        if len(sequences) == 0:
            continue

        # 示例检查
        sce_example = coden1_pad(sequences[0])
        print(f"SCE 编码形状: {sce_example.shape}")

        # 执行编码
        sce_features = []
        for seq in tqdm(sequences, desc=f"{split} encoding"):
            sce_features.append(coden1_pad(seq, max_len=99))

        # 保存 npz
        save_npz = os.path.join(dst_dir, f"{split}_sce.npz")
        np.savez_compressed(save_npz, X=np.array(sce_features, dtype=object), y=np.array(labels))
        print(f"已保存 SCE 编码到: {save_npz}")

        # 保存 pickle
        save_pkl = os.path.join(dst_dir, f"{split}_sce.pkl")
        with open(save_pkl, "wb") as f:
            pickle.dump({"X": sce_features, "y": labels}, f)
        print(f"已保存备份 pickle: {save_pkl}")