import datetime
import os
import cv2
from collections import defaultdict


class MotionLogger:
    def __init__(self):
        self.logs_dir = "logs"
        self.current_log_file = None
        self.object_counter = defaultdict(int)
        self.object_tracker = defaultdict(set)
        self.log_entry_count = 0

        os.makedirs(self.logs_dir, exist_ok=True)
        self._update_log_file()

    # ================== ВСПОМОГАТЕЛЬНЫЕ ==================

    def _update_log_file(self):
        """Создает новый лог-файл на каждый день"""
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        new_log_file = os.path.join(self.logs_dir, f"motion_log_{today}.txt")

        if new_log_file != self.current_log_file:
            self.current_log_file = new_log_file
            self.log_entry_count = 0
            self._print("[SYSTEM]", f"Новый файл лога: {self.current_log_file}", "system")

    def _make_log(self, tag, message):
        """Формирует запись в лог"""
        self._update_log_file()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] {tag} {message}\n"
        return entry, timestamp

    def _write_log(self, entry):
        """Записывает в файл"""
        try:
            with open(self.current_log_file, 'a', encoding='utf-8') as f:
                f.write(entry)
            self.log_entry_count += 1
        except Exception as e:
            self._print("[ERROR]", f"Ошибка записи лога: {e}", "error")

    def _print(self, tag, message, log_type="reset"):
        """Цветной вывод в терминал"""
        colors = {
            "system": "\033[96m",    # Голубой
            "camera": "\033[92m",    # Зеленый
            "motion": "\033[93m",    # Желтый
            "object": "\033[95m",    # Фиолетовый
            "settings": "\033[94m",  # Синий
            "trigger": "\033[33m",   # Оранжевый
            "error": "\033[91m",     # Красный
            "reset": "\033[0m"
        }
        print(f"{colors.get(log_type, colors['reset'])}{tag} {message}{colors['reset']}")

    # ================== ПУБЛИЧНЫЕ МЕТОДЫ ==================

    def log_system_event(self, message):
        entry, ts = self._make_log("[SYSTEM]", message)
        self._write_log(entry)
        self._print("[SYSTEM]", f"{ts}: {message}", "system")

    def log_camera_status(self, camera_idx, status):
        entry, ts = self._make_log(f"[CAM{camera_idx}]", status)
        self._write_log(entry)
        self._print(f"[CAM{camera_idx}]", f"{ts}: {status}", "camera")

    def log_motion_detected(self, camera_idx, is_triggered=False):
        if is_triggered:
            msg = f"Cam{camera_idx}: Камера включена по движению"
            entry, _ = self._make_log("[TRIGGER]", msg)
            self._print("[TRIGGER]", msg, "trigger")
        else:
            count = self.object_counter[camera_idx]
            msg = f"Cam{camera_idx}: Движение обнаружено (объектов: {count})"
            entry, _ = self._make_log("[MOTION]", msg)
            self._print("[MOTION]", msg, "motion")
        self._write_log(entry)

    def log_motion_stopped(self, camera_idx, duration, total_objects):
        msg = f"Cam{camera_idx}: Движение завершено (длительность: {duration:.1f}s, объектов: {total_objects})"
        entry, _ = self._make_log("[MOTION]", msg)
        self._write_log(entry)
        self._print("[MOTION]", msg, "motion")

    def log_new_objects(self, camera_idx, objects_info):
        for obj_id, obj_info in objects_info['new_objects'].items():
            msg = (f"Cam{camera_idx}: Новый объект #{obj_id} "
                   f"(позиция: {obj_info['position']}, размер: {obj_info['size'][0]}x{obj_info['size'][1]})")
            entry, _ = self._make_log("[OBJECT]", msg)
            self._write_log(entry)
            self._print("[OBJECT]", f"Cam{camera_idx}: Новый объект {obj_id}", "object")

    def log_motion_summary(self, camera_idx, objects_info):
        msg = (f"Cam{camera_idx}: Всего объектов: {objects_info['total_objects']}, "
               f"Активных: {objects_info['active_objects']}, "
               f"Новых: {len(objects_info['new_objects'])}, "
               f"Потерянных: {len(objects_info['lost_objects'])}")
        entry, _ = self._make_log("[SUMMARY]", msg)
        self._write_log(entry)
        # Сводка выводится только в файл

    def log_settings(self, settings):
        settings_str = ", ".join(f"{k}: {v}" for k, v in settings.items())
        entry, _ = self._make_log("[SETTINGS]", f"Настройки системы: {settings_str}")
        self._write_log(entry)
        self._print("[SETTINGS]", "Настройки системы сохранены", "settings")

    def log_error(self, message):
        entry, ts = self._make_log("[ERROR]", message)
        self._write_log(entry)
        self._print("[ERROR]", f"{ts}: {message}", "error")

    # ================== ТРЕКИНГ ОБЪЕКТОВ ==================

    def track_objects(self, camera_idx, contours, grid_size=10):
        """
        Трекинг объектов и подсчет:
        ID формируется по центру bbox (округляется по grid_size пикселей)
        """
        current_objects = set()
        objects_info = {
            'new_objects': {},
            'lost_objects': set(),
            'active_objects': len(contours),
            'total_objects': self.object_counter[camera_idx]
        }

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = x + w // 2, y + h // 2
            object_id = f"{camera_idx}_{cx//grid_size}_{cy//grid_size}"

            current_objects.add(object_id)
            if object_id not in self.object_tracker[camera_idx]:
                self.object_counter[camera_idx] += 1
                objects_info['new_objects'][object_id] = {
                    'position': (x, y),
                    'size': (w, h)
                }

        # Потерянные объекты
        objects_info['lost_objects'] = self.object_tracker[camera_idx] - current_objects
        self.object_tracker[camera_idx] = current_objects
        return objects_info

    def reset_camera_objects(self, camera_idx):
        self.object_counter[camera_idx] = 0
        self.object_tracker[camera_idx] = set()

    # ================== УПРАВЛЕНИЕ ЛОГАМИ ==================

    def cleanup_old_logs(self, days_to_keep=30):
        """Удаляет логи старше N дней"""
        try:
            current_time = datetime.datetime.now()
            for filename in os.listdir(self.logs_dir):
                if filename.startswith("motion_log_") and filename.endswith(".txt"):
                    file_path = os.path.join(self.logs_dir, filename)
                    file_time = datetime.datetime.fromtimestamp(os.path.getctime(file_path))
                    if (current_time - file_time).days > days_to_keep:
                        os.remove(file_path)
                        self.log_system_event(f"Удален старый лог-файл: {filename}")
        except Exception as e:
            self.log_error(f"Ошибка при очистке старых логов: {e}")


# Глобальный экземпляр
motion_logger = MotionLogger()
