import json
import pandas as pd
from tqdm import tqdm
import torch
from collections import Counter, defaultdict
import os


def read_original_file(file_name):
    """讀取原始數據文件，轉換為以id為key的格式"""
    labeled_data = {}
    with open(file_name, 'r', encoding='UTF-8-sig') as file:
        data = json.load(file)
        for item in data:
            comment = item['comment']
            emotions = item['emotions']
            id = item['id']
            segmented = item['segmented']

            emotions = [e for e in emotions if e not in ["非自我情緒", "諷刺"]]

            if emotions and "未標記" not in emotions:
                labeled_data[id] = {
                    "comment": comment,
                    "emotions": emotions,
                    "segmented": segmented
                }
    return labeled_data


def create_word_to_index(data):
    """建立詞彙表，不使用位置信息"""
    word_to_index = {}
    current_idx = 0

    for item_data in data.values():
        for word in item_data['segmented']:
            if word not in word_to_index:
                word_to_index[word] = current_idx
                current_idx += 1

    return word_to_index


def text_to_vector(segmented, word_to_index):
    """將segmented列表轉換為詞頻向量"""
    vector = torch.zeros(len(word_to_index), dtype=torch.float32)

    # 計算詞頻
    word_counts = Counter(segmented)

    # 填充向量
    for word, count in word_counts.items():
        if word in word_to_index:
            vector[word_to_index[word]] = count

    return vector, len(segmented)


def prepare_batch_data(data, word_to_index, device):
    """準備批次數據"""
    vectors = []
    lengths = []
    ids = []

    for id, item_data in tqdm(data.items(), desc="Converting texts to vectors"):
        vector, length = text_to_vector(item_data['segmented'], word_to_index)
        vectors.append(vector)
        lengths.append(length)
        ids.append(id)

    vectors = torch.stack(vectors).to(device)
    lengths = torch.tensor(lengths, dtype=torch.float32).to(device)
    return vectors, lengths, ids



def group_data_by_emotion(data):
    """將數據按主要情緒分組"""
    emotion_groups = defaultdict(list)
    for id, item_data in data.items():
        primary_emotion = item_data['emotions'][0]
        emotion_groups[primary_emotion].append((id, item_data))
    return emotion_groups


def distribute_emotion_data(emotion_data, similar_pairs, n_partitions):
    """將同一情緒的數據分配到不同分割區"""
    partitions = [{} for _ in range(n_partitions)]
    used_ids = set()

    # 建立ID到數據的映射
    id_to_data = {id: item_data for id, item_data in emotion_data}

    print("\n分配相似資料到不同分割區:")
    with tqdm(total=len(emotion_data), desc="Distributing data") as pbar:
        # 依序處理每個similar pairs組
        current_partition = 0
        processed_source_ids = set()

        for id1, id2, similarity in similar_pairs:
            # 如果這是一個新的源ID且還沒被使用過
            if id1 not in processed_source_ids and id1 not in used_ids:
                # 將源ID放入當前partition
                if id1 in id_to_data:
                    partitions[current_partition][id1] = id_to_data[id1]
                    used_ids.add(id1)
                    processed_source_ids.add(id1)
                    current_partition = (current_partition + 1) % n_partitions

                # 找出所有與此ID相似的配對
                related_pairs = [(i1, i2, sim) for i1, i2, sim in similar_pairs
                                 if (i1 == id1 or i2 == id1) and sim > 0.7]

                # 按相似度排序
                related_pairs.sort(key=lambda x: x[2], reverse=True)

                # 依序將相似的ID放入後續partition
                for rel_id1, rel_id2, _ in related_pairs:
                    target_id = rel_id2 if rel_id1 == id1 else rel_id1
                    if target_id not in used_ids and target_id in id_to_data:
                        partitions[current_partition][target_id] = id_to_data[target_id]
                        used_ids.add(target_id)
                        current_partition = (current_partition + 1) % n_partitions

            pbar.update(1)

    print("\n分配剩餘資料:")
    remaining_data = [(id, item_data) for id, item_data in emotion_data if id not in used_ids]

    with tqdm(total=len(remaining_data), desc="Distributing remaining data") as pbar:
        for id, item_data in remaining_data:
            min_partition = min(range(n_partitions), key=lambda x: len(partitions[x]))
            partitions[min_partition][id] = item_data
            pbar.update(1)

    return partitions


def save_partition(partition_data, index, output_dir):
    """保存單個partition到JSON文件"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, f'partition_{index}.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(partition_data, f, ensure_ascii=False, indent=2)
    return output_file


def save_partitions_to_excel(partition_files, excel_output_file):
    """將每個partition的情況保存到Excel文件"""
    partition_info = []

    for i, file_path in enumerate(partition_files):
        with open(file_path, 'r', encoding='utf-8') as f:
            partition_data = json.load(f)
            emotion_distribution = {}
            for item_data in partition_data.values():
                primary_emotion = item_data['emotions'][0]
                emotion_distribution[primary_emotion] = emotion_distribution.get(primary_emotion, 0) + 1

            partition_info.append({
                "Partition": i + 1,
                "Size": len(partition_data),
                **emotion_distribution
            })

    df = pd.DataFrame(partition_info)
    df.to_excel(excel_output_file, index=False)
    print(f"各partition的情況已保存到: {excel_output_file}")


def merge_partitions(partition_files, output_file):
    """合併所有partition文件到最終的訓練和測試集"""
    train_data = {}
    test_data = {}
    emotion_distribution = {}

    for i, file_path in enumerate(partition_files):
        with open(file_path, 'r', encoding='utf-8') as f:
            partition_data = json.load(f)
            if i < 9:  # 前9個partition作為訓練集
                train_data.update(partition_data)
            else:  # 最後1個partition作為測試集
                test_data.update(partition_data)

            # 更新情緒分布
            for item_data in partition_data.values():
                primary_emotion = item_data['emotions'][0]
                emotion_distribution[primary_emotion] = emotion_distribution.get(primary_emotion, 0) + 1

    output_data = {
        "train_data": train_data,
        "test_data": test_data
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    return {
        "train_size": len(train_data),
        "test_size": len(test_data),
        "split_ratio": len(train_data) / (len(train_data) + len(test_data)) if len(train_data) + len(
            test_data) > 0 else 0,
        "emotion_distribution": emotion_distribution
    }

def save_similarities_to_json(similar_pairs, emotion, output_file):
    """保存相似度計算結果到JSON檔案"""
    similarity_data = {
        "emotion": emotion,
        "similar_pairs": [
            {
                "id1": id1,
                "id2": id2,
                "similarity": similarity
            }
            for id1, id2, similarity in similar_pairs
        ]
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(similarity_data, f, ensure_ascii=False, indent=2)


def calculate_emotion_group_similarities(emotion_data, vectors, lengths, emotion_name, output_dir, batch_size=1000):
    """計算同一情緒組內的文本相似度，並保存結果"""
    n = len(emotion_data)
    device = vectors.device
    similar_pairs = []
    total_pairs = n * (n - 1) // 2

    if total_pairs == 0:
        return []

    print(f"\n開始計算相似度，總共需要處理 {total_pairs} 個配對...")

    # 正規化向量
    normalized_vectors = vectors / (vectors.norm(dim=1, keepdim=True) + 1e-8)

    with tqdm(total=total_pairs, desc="Calculating similarities") as pbar:
        for i in range(0, n, batch_size):
            torch.cuda.empty_cache()
            batch_end = min(i + batch_size, n)
            batch_vectors = normalized_vectors[i:batch_end].to(device)

            for j in range(i, n, batch_size):
                inner_batch_end = min(j + batch_size, n)
                inner_batch_vectors = normalized_vectors[j:inner_batch_end].to(device)

                # 計算餘弦相似度
                cosine_sim = torch.mm(batch_vectors, inner_batch_vectors.t())

                # 計算 Dice 係數
                overlap = torch.mm(vectors[i:batch_end], vectors[j:inner_batch_end].t()) * 2
                total_lengths = lengths[i:batch_end].unsqueeze(1) + lengths[j:inner_batch_end].unsqueeze(0)
                dice_sim = overlap / total_lengths

                # 結合兩種相似度
                similarity_matrix = (cosine_sim + dice_sim) / 2

                # 只保留上三角矩陣的值
                if i == j:
                    mask = torch.triu(torch.ones_like(similarity_matrix), diagonal=1)
                else:
                    mask = torch.ones_like(similarity_matrix)

                masked_similarities = similarity_matrix * mask

                # 找出相似度大於閾值的配對
                above_threshold = torch.nonzero(masked_similarities > 0.7)

                if above_threshold.size(0) > 0:
                    batch_similarities = masked_similarities[above_threshold[:, 0], above_threshold[:, 1]]

                    for idx, (row, col) in enumerate(above_threshold.cpu().numpy()):
                        id1 = emotion_data[int(row + i)][0]
                        id2 = emotion_data[int(col + j)][0]
                        similar_pairs.append((
                            id1,
                            id2,
                            float(batch_similarities[idx].cpu())
                        ))

                del inner_batch_vectors
                torch.cuda.empty_cache()

            del batch_vectors
            torch.cuda.empty_cache()

            pairs_in_batch = (batch_end - i) * (n - (i + batch_end) // 2)
            pbar.update(pairs_in_batch)

    # 按相似度排序
    similar_pairs = sorted(similar_pairs, key=lambda x: x[2], reverse=True)

    # 保存相似度結果
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f'similarities_{emotion_name}.json')
    save_similarities_to_json(similar_pairs, emotion_name, output_file)

    return similar_pairs



def split_and_save_data(input_file, output_dir, final_output_file, excel_output_file, similarities_dir):
    """主要處理函數"""
    print("讀取資料...")
    labeled_data = read_original_file(input_file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")

    print("\n建立詞彙表...")
    word_to_index = create_word_to_index(labeled_data)
    print(f"詞彙表大小: {len(word_to_index)}")

    print("\n準備向量數據...")
    vectors, lengths, ids = prepare_batch_data(labeled_data, word_to_index, device)

    print("\n開始按情緒分配數據...")
    emotion_groups = group_data_by_emotion(labeled_data)

    partitions = [{}] * 10
    for emotion, emotion_data in emotion_groups.items():
        print(f"\n處理情緒 '{emotion}' (共 {len(emotion_data)} 筆資料)")
        similar_pairs = calculate_emotion_group_similarities(
            list(emotion_data),  # Convert to list
            vectors,
            lengths,
            emotion,
            similarities_dir
        )
        print(f"找到 {len(similar_pairs)} 個相似配對")

        emotion_partitions = distribute_emotion_data(emotion_data, similar_pairs, 10)
        for i in range(10):
            if not partitions[i]:
                partitions[i] = {}
            partitions[i].update(emotion_partitions[i])

    # 保存各個partition
    partition_files = []
    for i, partition in enumerate(partitions):
        file_path = save_partition(partition, i+1, output_dir)
        partition_files.append(file_path)

    # 保存Excel報告
    save_partitions_to_excel(partition_files, excel_output_file)

    # 合併並保存最終的訓練和測試集
    metadata = merge_partitions(partition_files, final_output_file)

    return metadata, partition_files

if __name__ == "__main__":
    input_file = 'output_results_ckip/combined_output.json'
    output_dir = 'partitions_ptt_test'
    final_output_file = 'train_test_split_ptt.json'
    excel_output_file = 'partitions_ptt_info.xlsx'
    similarities_dir = 'similarities_results'

    try:
        metadata, partition_files = split_and_save_data(
            input_file,
            output_dir,
            final_output_file,
            excel_output_file,
            similarities_dir
        )

        print(f"\n數據分割完成！")
        print(f"個別partition文件已保存在: {output_dir}/")
        print(f"訓練集大小: {metadata['train_size']}")
        print(f"測試集大小: {metadata['test_size']}")
        print(f"訓練集比例: {metadata['split_ratio']:.2f}")
        print(f"整體情緒分佈: ")
        for emotion, count in metadata['emotion_distribution'].items():
            print(f"  {emotion}: {count}")
        print(f"最終結果已保存至: {final_output_file}")
        print(f"各partition的詳細資訊已保存至: {excel_output_file}")

    except Exception as e:
        print(f"處理過程中發生錯誤: {str(e)}")





