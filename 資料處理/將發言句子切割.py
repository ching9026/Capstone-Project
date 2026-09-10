import re
import os
import json

def remove_https(text):
    # 移除句子裡有https或http開頭的內容
    return re.sub(r'https?://[^\s]+', '', text)

def has_repeated_characters(text, max_repeats=10):
    # 如果有洗版的先刪掉
    pattern = r'(.)\1{%d,}' % (max_repeats - 1)
    return re.search(pattern, text) is not None

def split_text(text, counters):
    text = text.replace("\n", "")
    text = remove_https(text)

    # 優先次序
    primary_punctuations = r'([.。？！])'
    secondary_punctuations = r'([,，；：;])'
    tertiary_punctuations = r'(\s+)'

    first_priority_segments = re.split(primary_punctuations, text)
    segments = []

    for i in range(len(first_priority_segments)):
        if i % 2 == 0:
            segment = first_priority_segments[i]

            if i > 0:
                counters['primary'] += 1

            second_priority_segments = re.split(secondary_punctuations, segment)
            for j in range(len(second_priority_segments)):
                if j % 2 == 0:
                    sub_segment = second_priority_segments[j]

                    if j > 0:
                        counters['secondary'] += 1

                    third_priority_segments = re.split(tertiary_punctuations, sub_segment)
                    for k in range(len(third_priority_segments)):
                        if third_priority_segments[k].strip():
                            if not has_repeated_characters(third_priority_segments[k].strip(), max_repeats=10):
                                segments.append(third_priority_segments[k].strip())

                            if k > 0:
                                counters['tertiary'] += 1
                else:
                    if segments:
                        segments[-1] += second_priority_segments[j]
        else:
            if segments:
                segments[-1] += first_priority_segments[i]

    return segments

input_dir = "結果檔/合併發言後加id"  # 銜接merged_content的output
output_dir = "結果檔/test"  # 拿來分析各情緒字的統計，並開始標記

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if not os.path.exists(input_dir):
    print(f"{input_dir} 不存在")
else:
    processed_article_ids = set()  # 用於跟踪已處理的 article_id

    for file_name in os.listdir(input_dir):
        if file_name.endswith(".json"):
            input_file_path = os.path.join(input_dir, file_name)
            comments = []
            counters = {'primary': 0, 'secondary': 0, 'tertiary': 0}

            try:
                with open(input_file_path, "r", encoding="utf-8-sig") as file:
                    content = file.read()
                    json_data = json.loads(content)
                    for article in json_data.get("articles", []):
                        article_id = article.get("article_id")

                        # 檢查 article_id 是否已經處理過
                        if article_id in processed_article_ids:
                            continue  # 跳過已處理的 article_id

                        processed_article_ids.add(article_id)  # 添加到已處理集合

                        for message in article.get("messages", []):
                            comment_content = message.get("merged_push_content", "")
                            push_index = message.get("push_index", "")
                            split_comment = split_text(comment_content, counters)

                            for idx, segment in enumerate(split_comment):
                                numbered_segment = f"{segment} {push_index}_{idx + 1}"
                                comments.append(numbered_segment)

                output_file_path = os.path.join(output_dir, f"{file_name}")
                with open(output_file_path, "w", encoding="utf-8") as output_file:
                    json.dump(comments, output_file, ensure_ascii=False, indent=4)

                print(f"已保存至 {output_file_path}")

            except FileNotFoundError:
                print(f"文件 {input_file_path} 不存在")
            except json.JSONDecodeError:
                print(f"文件 {input_file_path} 的 JSON格式不正確")
            except Exception as e:
                print(f"处理文件 {input_file_path} 出現錯誤: {e}")
