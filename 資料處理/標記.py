import json
import os
import re
import sys

import pygame

pygame.init()

screen_width = 1000
screen_height = 800
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("留言標記")

white = (255, 255, 255)
black = (0, 0, 0)
blue = (0, 0, 255)
green = (0, 255, 0)
red = (255, 0, 0)
yellow = (255, 255, 0)
purple = (128, 0, 128)
gray = (200, 200, 200)
shallow_blue = (0, 255, 255)
lime = (50, 205, 50)
orange = (255, 165, 0)

font_path = 'nesscary/Noto_Sans_TC/NotoSansTC-VariableFont_wght.ttf'
default_font_size = 24
font = pygame.font.Font(font_path, default_font_size)

input_dir = "暫時用不到/test5"
output_dir = "output_results1"

current_file_name = ""

LAST_MARKED_FILE = "last_marked.txt"


class Button:
    def __init__(self, text, position, size, color, font, action=None):
        self.text = text
        self.position = position
        self.size = size
        self.color = color
        self.font = font
        self.action = action
        self.rect = pygame.Rect(position, size)

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, self.rect)
        text_surface = self.font.render(self.text, True, black)
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)

    def is_clicked(self, mouse_pos):
        return self.rect.collidepoint(mouse_pos)


def get_buttons(comment_position, comment_height):
    button_size = (100, 40)
    button_x_start = 50
    button_y_position = comment_position[1] + comment_height + 10
    font_size = 20
    small_font = pygame.font.Font(font_path, font_size)

    button_spacing = button_size[0] + 10

    return [
        Button("愉悅", (button_x_start, button_y_position), button_size, green, small_font, action="joy"),
        Button("悲傷", (button_x_start + button_spacing, button_y_position), button_size, blue, small_font,
               action="sadness"),
        Button("憤怒", (button_x_start + 2 * button_spacing, button_y_position), button_size, red, small_font,
               action="anger"),
        Button("恐懼", (button_x_start + 3 * button_spacing, button_y_position), button_size, yellow, small_font,
               action="fear"),
        Button("厭惡", (button_x_start + 4 * button_spacing, button_y_position), button_size, purple, small_font,
               action="surprise"),
        Button("諷刺", (button_x_start + 5 * button_spacing, button_y_position), button_size, gray, small_font,
               action="Satire"),
        Button("看不出情緒", (button_x_start + 6 * button_spacing, button_y_position), button_size, lime, small_font,
               action="neutral"),
        Button("非自我情緒", (button_x_start + 7 * button_spacing, button_y_position), button_size, orange, small_font,
               action="non-self-emotion")
    ]


def draw_label_with_border(screen, text, position, font, text_color, bg_color, border_color, border_thickness,
                           max_width):
    wrapped_text = wrap_text(text, font, max_width)
    text_surfaces = [font.render(line, True, text_color) for line in wrapped_text]
    max_text_width = max(surface.get_width() for surface in text_surfaces)
    total_text_height = sum(surface.get_height() for surface in text_surfaces)

    label_rect = pygame.Rect(position,
                             (max_text_width + 2 * border_thickness, total_text_height + 2 * border_thickness))
    border_rect = label_rect.inflate(border_thickness, border_thickness)

    pygame.draw.rect(screen, border_color, border_rect)
    pygame.draw.rect(screen, bg_color, label_rect)

    y_offset = position[1] + border_thickness
    for surface in text_surfaces:
        screen.blit(surface, (position[0] + border_thickness, y_offset))
        y_offset += surface.get_height()

    return label_rect.height


def wrap_text(text, font, max_width):
    lines = []
    current_line = ""

    for char in text:
        test_line = current_line + char
        test_width, _ = font.size(test_line)

        if test_width <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = char

    if current_line:
        lines.append(current_line)

    return lines


def save_results(file_name, comments, comment_emotions, id_collect):
    results = []
    for comment, emotions, comment_id in zip(comments, comment_emotions, id_collect):
        result = {
            "comment": comment,
            "emotions": emotions,
            "id": comment_id
        }
        results.append(result)

    output_file_path = os.path.join(output_dir, file_name.replace('.json', '_emotions.json'))
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "w", encoding="utf-8") as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)
    print(f"結果已保存到 {output_file_path}")


def process_json_file(file_path):
    with open(file_path, "r", encoding="utf-8-sig") as file:
        data = json.load(file)
    return data


def check_existing_results(file_name):
    output_file_path = os.path.join(output_dir, file_name.replace('.json', '_emotions.json'))
    if os.path.exists(output_file_path):
        with open(output_file_path, "r", encoding="utf-8") as json_file:
            return json.load(json_file)
    return None


def get_last_marked_file():
    if os.path.exists(LAST_MARKED_FILE):
        with open(LAST_MARKED_FILE, "r") as f:
            return f.read().strip()
    return None


def save_last_marked_file(file_name):
    with open(LAST_MARKED_FILE, "w") as f:
        f.write(file_name)


def get_next_unmarked_file(input_dir, last_marked):
    json_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".json")])
    if last_marked:
        try:
            start_index = json_files.index(last_marked) + 1
        except ValueError:
            start_index = 0
    else:
        start_index = 0

    return json_files[start_index:]


def main():
    global current_file_name

    running = True

    last_marked = get_last_marked_file()
    json_files = get_next_unmarked_file(input_dir, last_marked)

    if not json_files:
        print("所有檔案都已經標記完成。")
        return

    file_index = 0
    id_collect = []

    prev_button = Button("上一頁", (350, 700), (100, 50), gray, font)
    next_button = Button("下一頁", (550, 700), (100, 50), gray, font)

    esc_pressed = False

    while running and file_index < len(json_files):
        file_name = json_files[file_index]
        file_path = os.path.join(input_dir, file_name)
        current_file_name = f"目前電影是: {file_name}"

        existing_results = check_existing_results(file_name)
        if existing_results:
            comments = [result["comment"] for result in existing_results]
            comment_emotions = [result["emotions"] for result in existing_results]
            id_collect = [result["id"] for result in existing_results]
            start_index = 0
        else:
            comments = process_json_file(file_path)
            id_collect.clear()

            for i in range(len(comments)):
                match = re.search(r"(\S+_\S+_\S+_\d+)$", comments[i])
                if match:
                    text_part = comments[i][:match.start() - 1]
                    text_part2 = comments[i][match.start():]
                    comments[i] = text_part
                    id_collect.append(text_part2)

            comment_emotions = [["未標記"] for _ in comments]
            start_index = 0

        total_pages = (len(comments) + 4) // 5
        current_page = (start_index // 5) + 1

        comment_buttons = {}

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = event.pos
                    if prev_button.is_clicked(mouse_pos):
                        if start_index > 0:
                            start_index -= 5
                            current_page -= 1
                    elif next_button.is_clicked(mouse_pos):
                        if start_index + 5 < len(comments):
                            start_index += 5
                            current_page += 1
                    for i, buttons in comment_buttons.items():
                        for button in buttons:
                            if button.is_clicked(mouse_pos):
                                if len(comment_emotions[i]) < 2:
                                    if button.text in comment_emotions[i]:
                                        comment_emotions[i].remove(button.text)
                                    else:
                                        if "未標記" in comment_emotions[i]:
                                            comment_emotions[i].remove("未標記")
                                        comment_emotions[i].append(button.text)
                                else:
                                    if button.text in comment_emotions[i]:
                                        comment_emotions[i].remove(button.text)

            keys = pygame.key.get_pressed()
            if keys[pygame.K_ESCAPE]:
                if not esc_pressed:
                    save_results(file_name, comments, comment_emotions, id_collect)
                    save_last_marked_file(file_name)  # 保存最後標記的文件名
                    file_index += 1
                    esc_pressed = True
                    break
            else:
                esc_pressed = False

            screen.fill(white)
            comment_buttons.clear()

            y_position = 100
            max_width = 800

            draw_text(screen, current_file_name, (screen_width / 2 - 200, 0), font, black)

            for i in range(start_index, min(start_index + 5, len(comments))):
                comment_height = draw_label_with_border(screen, f"{comments[i]} {comment_emotions[i]}",
                                                        (50, y_position), font, black, white, gray, 4, max_width)

                comment_buttons[i] = get_buttons((50, y_position), comment_height)
                for button in comment_buttons[i]:
                    button.draw(screen)
                y_position += comment_height + 60

            prev_button.draw(screen)
            next_button.draw(screen)

            page_info = f"頁碼: {current_page} / {total_pages}"
            draw_text(screen, page_info, (screen_width / 2 - 50, 750), font, black, 24)

            pygame.display.flip()

    pygame.quit()
    sys.exit()


def draw_text(surface, text, position, font, color, font_size=35):
    text_surface = font.render(text, True, color)
    text_surface = pygame.transform.scale(text_surface, (int(text_surface.get_width() * font_size / default_font_size),
                                                         int(text_surface.get_height() * font_size / default_font_size)))
    surface.blit(text_surface, position)


if __name__ == "__main__":
    main()