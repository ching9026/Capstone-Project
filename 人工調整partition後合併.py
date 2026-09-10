import json
import pandas as pd
from pathlib import Path
from collections import defaultdict


def analyze_partition_emotions(partition_data):
    """分析單個partition的情緒分布"""
    emotions = defaultdict(int)
    for item_data in partition_data.values():
        primary_emotion = item_data['emotions'][0]
        emotions[primary_emotion] += 1
    return dict(emotions)


def create_emotion_distribution_excel(partition_emotions, output_file):
    """將情緒分布數據轉換為Excel格式

    Args:
        partition_emotions (list): 各partition的情緒分布數據
        output_file (str): Excel輸出文件路徑
    """
    # 獲取所有情緒類別
    all_emotions = set()
    for partition in partition_emotions:
        all_emotions.update(partition['emotions'].keys())
    all_emotions = sorted(list(all_emotions))

    # 準備DataFrame數據
    data = []
    for partition in partition_emotions:
        row = {
            'Partition': f'Partition {partition["partition"]}',
            'Size': partition['size']
        }
        # 添加各情緒數量
        for emotion in all_emotions:
            row[emotion] = partition['emotions'].get(emotion, 0)

        data.append(row)

    # 創建DataFrame
    columns = ['Partition', 'Size'] + all_emotions
    df = pd.DataFrame(data, columns=columns)

    # 保存為Excel
    df.to_excel(output_file, index=False)


def merge_partitions(partition_files, output_file, excel_output):
    """合併所有partition文件到最終的訓練和測試集，並生成Excel報告"""
    train_data = {}
    test_data = {}
    emotion_distribution = {}
    partition_emotions = []

    # 確保輸出目錄存在
    for file_path in [Path(output_file), Path(excel_output)]:
        file_path.parent.mkdir(parents=True, exist_ok=True)

    # 讀取並合併數據
    for i, file_path in enumerate(partition_files):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                partition_data = json.load(f)

                partition_emotion_dist = analyze_partition_emotions(partition_data)
                partition_emotions.append({
                    'partition': i + 1,
                    'size': len(partition_data),
                    'emotions': partition_emotion_dist
                })

                if i < 9:  # 前9個partition作為訓練集
                    train_data.update(partition_data)
                else:  # 最後1個partition作為測試集
                    test_data.update(partition_data)

                for emotion, count in partition_emotion_dist.items():
                    emotion_distribution[emotion] = emotion_distribution.get(emotion, 0) + count

        except FileNotFoundError:
            print(f"警告：找不到文件 {file_path}")
            continue
        except json.JSONDecodeError:
            print(f"警告：文件 {file_path} 不是有效的JSON格式")
            continue

    # 保存合併後的數據
    output_data = {"train_data": train_data, "test_data": test_data}
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存JSON文件時發生錯誤: {str(e)}")
        raise

    # 創建Excel報告
    try:
        create_emotion_distribution_excel(partition_emotions, excel_output)
    except Exception as e:
        print(f"保存Excel文件時發生錯誤: {str(e)}")
        raise

    # 返回統計信息
    total_samples = len(train_data) + len(test_data)
    metadata = {
        "train_size": len(train_data),
        "test_size": len(test_data),
        "split_ratio": len(train_data) / total_samples if total_samples > 0 else 0,
        "emotion_distribution": emotion_distribution
    }

    return metadata, partition_emotions


if __name__ == "__main__":
    # 配置
    output_dir = 'partitions_ptt'
    final_output_file = 'train_test_split_ptt_test.json'
    excel_output_file = 'emotion_distribution.xlsx'

    # 構建partition文件路徑列表
    partition_files = [f"{output_dir}/partition_{i + 1}.json" for i in range(10)]

    try:
        # 執行合併並生成報告
        metadata, partition_emotions = merge_partitions(partition_files, final_output_file, excel_output_file)

        # 輸出整體結果
        print(f"\n數據合併完成！")
        print(f"訓練集大小: {metadata['train_size']}")
        print(f"測試集大小: {metadata['test_size']}")
        print(f"訓練集比例: {metadata['split_ratio']:.2f}")
        print(f"\n整體情緒分佈: ")
        for emotion, count in metadata['emotion_distribution'].items():
            print(f"  {emotion}: {count}")

        print(f"\n詳細的情緒分布報告已保存至: {excel_output_file}")
        print(f"合併後的數據已保存至: {final_output_file}")

    except Exception as e:
        print(f"處理過程中發生錯誤: {str(e)}")