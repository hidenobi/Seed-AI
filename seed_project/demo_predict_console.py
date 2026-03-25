import curses
from pathlib import Path

from predict import predict_single_image


DEMO_DIR = Path("demo_images")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def load_demo_images():
    if not DEMO_DIR.exists():
        return []

    return sorted(
        [path for path in DEMO_DIR.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    )


def draw_screen(stdscr, image_paths, selected_idx, scroll_offset, result_lines, command_buffer, status_line):
    stdscr.erase()
    height, width = stdscr.getmaxyx()

    title = "Demo Prediction Console"
    instructions = "Up/Down: chon anh | Enter: du doan | :q: thoat"
    stdscr.addnstr(0, 0, title, width - 1, curses.A_BOLD)
    stdscr.addnstr(1, 0, instructions, width - 1)

    if not image_paths:
        stdscr.addnstr(3, 0, f"Khong tim thay anh trong {DEMO_DIR}", width - 1)
        stdscr.refresh()
        return

    list_top = 3
    list_height = max(5, height - 10)

    for row in range(list_height):
        image_idx = scroll_offset + row
        if image_idx >= len(image_paths):
            break

        image_name = image_paths[image_idx].name
        prefix = "> " if image_idx == selected_idx else "  "
        line = f"{prefix}{image_idx + 1:02d}. {image_name}"
        attr = curses.A_REVERSE if image_idx == selected_idx else curses.A_NORMAL
        stdscr.addnstr(list_top + row, 0, line, width - 1, attr)

    result_top = list_top + list_height + 1
    divider = "-" * max(1, width - 1)
    if result_top < height:
        stdscr.addnstr(result_top, 0, divider, width - 1)

    for idx, line in enumerate(result_lines, start=1):
        row = result_top + idx
        if row >= height - 2:
            break
        stdscr.addnstr(row, 0, line, width - 1)

    command_line = f"Command: {command_buffer}" if command_buffer else "Command: "
    stdscr.addnstr(height - 2, 0, command_line, width - 1)
    stdscr.addnstr(height - 1, 0, status_line, width - 1)
    stdscr.refresh()


def run_prediction(selected_image):
    result = predict_single_image(str(selected_image), verbose=False)
    if result is None:
        return [
            f"Khong the du doan anh: {selected_image.name}",
        ]

    return [
        f"Anh da chon: {selected_image.name}",
        f"Nhan du doan: {result['label']}",
        f"Do tin cay: {result['confidence']:.2f}%",
    ]


def main(stdscr):
    try:
        curses.curs_set(0)
    except curses.error:
        pass
    stdscr.keypad(True)

    image_paths = load_demo_images()
    selected_idx = 0
    scroll_offset = 0
    result_lines = ["Chon mot anh va nhan Enter de chay du doan."]
    command_buffer = ""
    status_line = f"Tong so anh demo: {len(image_paths)}"

    while True:
        height, _ = stdscr.getmaxyx()
        list_height = max(5, height - 10)

        if image_paths:
            selected_idx = max(0, min(selected_idx, len(image_paths) - 1))

            if selected_idx < scroll_offset:
                scroll_offset = selected_idx
            elif selected_idx >= scroll_offset + list_height:
                scroll_offset = selected_idx - list_height + 1

        draw_screen(stdscr, image_paths, selected_idx, scroll_offset, result_lines, command_buffer, status_line)
        key = stdscr.getch()

        if command_buffer:
            if key in (10, 13):
                if command_buffer == ":q":
                    break
                status_line = f"Lenh khong hop le: {command_buffer}"
                command_buffer = ""
                continue

            if key in (27,):
                command_buffer = ""
                status_line = "Da huy command mode."
                continue

            if key in (curses.KEY_BACKSPACE, 127, 8):
                command_buffer = command_buffer[:-1]
                continue

            if 32 <= key <= 126:
                command_buffer += chr(key)
                continue

            continue

        if key == curses.KEY_UP and image_paths:
            selected_idx -= 1
            status_line = f"Dang chon: {image_paths[selected_idx].name}"
            continue

        if key == curses.KEY_DOWN and image_paths:
            selected_idx += 1
            status_line = f"Dang chon: {image_paths[selected_idx].name}"
            continue

        if key in (10, 13) and image_paths:
            selected_image = image_paths[selected_idx]
            status_line = f"Dang du doan: {selected_image.name}"
            draw_screen(
                stdscr, image_paths, selected_idx, scroll_offset, result_lines, command_buffer, status_line
            )
            result_lines = run_prediction(selected_image)
            status_line = "Da du doan xong. Tiep tuc chon anh khac hoac go :q de thoat."
            continue

        if key == ord(":"):
            command_buffer = ":"
            status_line = "Nhap command, Enter de xac nhan."
            continue

        if key in (ord("q"), ord("Q")):
            status_line = "De thoat, hay go :q va nhan Enter."
            continue


if __name__ == "__main__":
    curses.wrapper(main)
