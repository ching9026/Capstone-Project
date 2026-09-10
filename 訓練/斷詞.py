import os
import json
from ckiptagger import WS
import torch


def main():
    input_dir = "merge_after"
    output_dir = "output_results_ckip"

    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)

    # 檢查 GPU 是否可用
    device = 0 if torch.cuda.is_available() else -1
    if device == 0:
        print("使用 GPU 運行")
    else:
        print("GPU 不可用，使用 CPU 運行")

    # 初始化 WS 模型，指定使用 GPU
    ws = WS("./nesscary/data", disable_cuda=False)

    # 設置批次大小，根據您的 GPU 內存大小調整
    batch_size = 32

    # 遍歷輸入目錄中的每個文件
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".json"):
            input_file_path = os.path.join(input_dir, file_name)
            print(f"處理文件: {file_name}")

            with open(input_file_path, "r", encoding="utf-8-sig") as file:
                content = file.read()
                json_data = json.loads(content)

                # 批次處理數據
                processed_data = []
                comments_batch = []
                current_batch = []

                for i, record in enumerate(json_data):
                    current_batch.append(record)
                    comments_batch.append(record['comment'])

                    # 當達到批次大小或是最後一條數據時進行處理
                    if len(comments_batch) == batch_size or i == len(json_data) - 1:
                        try:
                            # 使用 WS 模型進行分句處理
                            segmented_sentences = ws(comments_batch)

                            # 將分句結果添加到對應的記錄中
                            for record, segmented in zip(current_batch, segmented_sentences):
                                record['segmented'] = segmented
                                processed_data.append(record)

                            # 清空批次
                            comments_batch = []
                            current_batch = []

                            # 清理 GPU 緩存
                            if device == 0:
                                torch.cuda.empty_cache()

                        except Exception as e:
                            print(f"處理批次時出錯: {str(e)}")
                            # 如果出錯，可以嘗試減小批次大小重試
                            if batch_size > 1:
                                batch_size = batch_size // 2
                                print(f"減小批次大小至: {batch_size}")
                                # 將當前批次的數據放回待處理隊列
                                comments_batch = []
                                current_batch = []
                            else:
                                raise e

            # 輸出結果到新文件
            output_file_path = os.path.join(output_dir, file_name)
            with open(output_file_path, "w", encoding='utf-8-sig') as json_file:
                json.dump(processed_data, json_file, ensure_ascii=False, indent=4)

            print(f"完成處理文件: {file_name}")

    # 釋放資源
    del ws
    if device == 0:
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # 設置 GPU 內存分配策略
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.8)  # 使用 80% 的 GPU 內存
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

    try:
        main()
        print("所有文件處理完成")
    except Exception as e:
        print(f"程序執行出錯: {str(e)}")