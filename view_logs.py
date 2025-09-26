#!/usr/bin/env python3
import os

def view_logs():
    logs_dir = "logs"
    
    if not os.path.exists(logs_dir):
        print("Папка с логами не найдена!")
        return
    
    # Получаем список файлов логов
    log_files = [f for f in os.listdir(logs_dir) if f.startswith("motion_log_") and f.endswith(".txt")]
    log_files.sort(reverse=True)
    
    if not log_files:
        print("Лог-файлы не найдены!")
        return
    
    print("Доступные лог-файлы:")
    for i, file in enumerate(log_files, 1):
        print(f"{i}. {file}")
    
    try:
        choice = int(input("\nВыберите файл для просмотра (0 для выхода): "))
        if choice == 0:
            return
        
        selected_file = log_files[choice - 1]
        file_path = os.path.join(logs_dir, selected_file)
        
        # Показываем содержимое файла
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"\nСодержимое файла {selected_file}:")
        print("=" * 80)
        print(content)
        print("=" * 80)
        
    except (ValueError, IndexError):
        print("Неверный выбор!")

if __name__ == "__main__":
    view_logs()