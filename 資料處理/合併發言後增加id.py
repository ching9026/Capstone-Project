import os
import json
from collections import defaultdict

input_dir = "結果檔/合併發言"  # From merge_content(new) output
output_dir = "結果檔/合併發言後加id"  # Used later for sentence segmentation in split.py

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

arc_index = 0  # Initialize the arc_index for numbering articles across files
previous_article_id = None  # Track previous article_id

for file_name in os.listdir(input_dir):
    index = 1  # Initialize the index for numbering merged contents
    if file_name.endswith(".json"):
        input_file_path = os.path.join(input_dir, file_name)

        with open(input_file_path, "r", encoding="utf-8") as file:
            content = file.read()
            data = json.loads(content)
            all_articles = data["articles"]

        for article in all_articles:
            if article.get('article_id') != previous_article_id:
                arc_index += 1  # Increment arc_index if article_id changes
                previous_article_id = article.get('article_id')
            article['arc_index'] = arc_index

            merged_messages = defaultdict(lambda: {'push_tag': None, 'push_contents': []})
            if "messages" in article:
                for message in article["messages"]:
                    key = (message['push_ipdatetime'], message['push_userid'])
                    if merged_messages[key]['push_tag'] is None:
                        merged_messages[key]['push_tag'] = message.get('push_tag')
                    merged_messages[key]['push_contents'].append(
                        message.get('push_content', ''))
            else:
                article["messages"] = []

            new_messages = []
            for key, value in merged_messages.items():
                push_ipdatetime, push_userid = key
                merged_push_content = ''.join(value['push_contents'])
                new_filename = file_name.replace(".json", "")
                new_message = {
                    'merged_push_content': ''.join(value['push_contents']),
                    'push_ipdatetime': push_ipdatetime,
                    'push_tag': value['push_tag'],
                    'push_userid': push_userid,
                    'push_index': f"{new_filename}_P{arc_index}_C{index}"
                }
                new_messages.append(new_message)
                index += 1  # Increment the index for the next merged content

            article["messages"] = new_messages


            index = 1  # Reset index for the next article
        arc_index = 0
        output_file_name = file_name
        output_file_path = os.path.join(output_dir, output_file_name)

        with open(output_file_path, "w", encoding="utf-8") as output_file:
            json.dump({"articles": all_articles}, output_file, ensure_ascii=False, indent=4)
