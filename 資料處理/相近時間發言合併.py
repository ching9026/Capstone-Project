import json
import os
from datetime import datetime, timedelta

input_dir = "結果檔/ptt原始資料" # 從movie_rename的output拿來
output_dir = "結果檔/合併發言"  # 銜接到merged_content

# 确保输出目录存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 定义日期时间格式
datetime_format = "%m/%d %H:%M"

# 遍历输入目录中的所有 JSON 文件
for file_name in os.listdir(input_dir):
    if file_name.endswith(".json"):
        input_file_path = os.path.join(input_dir, file_name)
        output_file_path = os.path.join(output_dir, file_name)

        with open(input_file_path, "r", encoding="utf-8") as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                print(f"文件 {file_name} 不是有效的 JSON 文件，跳过...")
                continue

        # 处理每一篇文章
        for article in data.get('articles', []):
            messages = article.get('messages', [])
            new_messages = []
            user_msgs = {}

            # 按用户整理消息
            for msg in messages:
                push_ipdatetime = msg.get('push_ipdatetime', '')
                push_tag = msg.get('push_tag', '')  # 获取 push_tag 字段

                user = msg['push_userid']
                if user not in user_msgs:
                    user_msgs[user] = []
                user_msgs[user].append(msg)

            for user, msgs in user_msgs.items():
                buffer = []
                prev_time = None

                for i, msg in enumerate(msgs):
                    push_ipdatetime = msg.get('push_ipdatetime', '')
                    push_tag = msg.get('push_tag', '')  # 获取 push_tag 字段

                    try:
                        current_time = datetime.strptime(push_ipdatetime, datetime_format)
                    except ValueError:
                        print(f"无法解析的时间格式: {push_ipdatetime}, 文件: {file_name}")
                        continue

                    if i == 0:
                        buffer.append(msg['push_content'])
                        prev_time = current_time
                        continue

                    time_diff = current_time - prev_time

                    if time_diff <= timedelta(minutes=1) and len(buffer[-1]) >= 23:
                        buffer.append(msg['push_content'])
                    else:
                        new_messages.append({
                            'push_userid': user,
                            'push_content': ''.join(buffer),
                            'push_ipdatetime': msgs[i - 1]['push_ipdatetime'],
                            'push_tag': msgs[i - 1].get('push_tag', '')  # 保留 push_tag 字段
                        })
                        buffer = [msg['push_content']]

                    prev_time = current_time

                if buffer:
                    new_messages.append({
                        'push_userid': user,
                        'push_content': ''.join(buffer),
                        'push_ipdatetime': msgs[-1]['push_ipdatetime'],
                        'push_tag': msgs[-1].get('push_tag', '')  # 保留 push_tag 字段
                    })

            # 拆分长内容
            final_messages = []
            for msg in new_messages:
                content = msg['push_content']
                # 以特定条件拆分，例如句号或特定长度
                if len(content) > 100:  # 可以根据需要调整条件
                    parts = [content[i:i + 100] for i in range(0, len(content), 100)]
                else:
                    parts = [content]

                for part in parts:
                    new_msg = msg.copy()
                    new_msg['push_content'] = part.strip()
                    final_messages.append(new_msg)

            # 更新文章的消息列表
            article['messages'] = final_messages

        # 保存合并后的结果到新的 JSON 文件
        with open(output_file_path, "w", encoding="utf-8") as output_file:
            json.dump(data, output_file, ensure_ascii=False, indent=4)
