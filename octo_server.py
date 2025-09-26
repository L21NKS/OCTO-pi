import socket
import struct
import pickle
import cv2
import json
import threading
import numpy as np
import time
import os
import sys
import subprocess

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from motion_detection import detect_motion, draw_motion_visualization
from face_detection import load_face_detection_model, detect_faces
from camera_utils import (
    initialize_cameras, release_cameras, create_video_grid,
    get_no_signal_frame, get_waiting_frame
)
from logger import motion_logger

def get_local_ip():
    """Автоматическое определение IP адреса"""
    try:
        # Создаем временное соединение чтобы узнать свой IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "0.0.0.0"

# Используем автоматическое определение IP
HOST = get_local_ip()
PORT_VIDEO = 9999
PORT_CMD = 9998

print("=" * 50)
print("OCTO Surveillance Server")
print("=" * 50)
print(f"Server IP: {HOST}")
print(f"Video port: {PORT_VIDEO}")
print(f"Command port: {PORT_CMD}")
print("=" * 50)

class HeadlessSurveillanceSystem:
    def __init__(self):
        # Пробуем разные индексы камер
        self.camera_indices = self.detect_cameras()
        self.caps = []
        self.face_net = None
        self.masks = {}

        # Состояние камер
        self.motion_detected = {}
        self.prev_frames = {}
        self.last_motion_time = {}
        self.last_motion_check = {}
        self.motion_start_time = {}
        self.motion_contours = {}
        self.last_check_time = {}

        for cam_idx in self.camera_indices:
            self.motion_detected[cam_idx] = False
            self.prev_frames[cam_idx] = None
            self.last_motion_time[cam_idx] = 0
            self.last_motion_check[cam_idx] = 0
            self.motion_start_time[cam_idx] = 0
            self.motion_contours[cam_idx] = []
            self.last_check_time[cam_idx] = 0

        self.camera_triggered = self.camera_indices[:]  # Все камеры по умолчанию
        self.camera_faces = self.camera_indices[:]
        self.camera_motion = self.camera_indices[:]
        self.MOTION_TIMEOUT = 30
        self.CHECK_INTERVAL = 1
        self.MOTION_THRESHOLD = 25
        self.MOTION_MIN_AREA = 500

        self.active_motion_cameras = set()

    def detect_cameras(self):
        """Обнаружение работающих камер"""
        working_cameras = []
        print("[SYSTEM] Поиск доступных камер...")
        
        for i in range(4):  # Проверяем камеры 0-3
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    working_cameras.append(i)
                    print(f"[SYSTEM] Камера {i} найдена - OK")
                else:
                    print(f"[SYSTEM] Камера {i} не возвращает кадры")
                cap.release()
            else:
                print(f"[SYSTEM] Камера {i} не открывается")
        
        if not working_cameras:
            print("[SYSTEM] Предупреждение: не найдено ни одной работающей камеры!")
            return [0]  # Все равно пробуем использовать камеру 0
        
        return working_cameras

    def initialize(self):
        """Инициализация системы"""
        motion_logger.log_system_event("Инициализация системы видеонаблюдения (HEADLESS)")

        # Загрузка модели детектирования лиц (пропускаем ошибку)
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            face_proto = os.path.join(script_dir, "opencv_face_detector.pbtxt")
            face_model = os.path.join(script_dir, "opencv_face_detector_uint8.pb")
            
            if os.path.exists(face_proto) and os.path.exists(face_model):
                self.face_net = load_face_detection_model(face_proto, face_model)
                motion_logger.log_system_event("Модель детектирования лиц загружена")
            else:
                motion_logger.log_system_event("Файлы модели лиц не найдены")
        except Exception as e:
            motion_logger.log_system_event(f"Ошибка загрузки модели лиц: {e}")

        # Инициализация камер
        self.caps = initialize_cameras(self.camera_indices)
        
        # Логирование настроек
        settings = {
            'working_cameras': self.camera_indices,
            'timeout': self.MOTION_TIMEOUT,
            'threshold': self.MOTION_THRESHOLD
        }
        motion_logger.log_settings(settings)

        motion_logger.log_system_event(f"Система инициализирована. Работающие камеры: {self.camera_indices}")

    def process_camera_frame(self, camera_idx, frame, current_time):
        """Обработка кадра камеры"""
        if frame is None:
            return get_no_signal_frame(camera_idx)

        frame = cv2.resize(frame, (640, 480))

        # Камера в режиме triggered + motion
        if camera_idx in self.camera_triggered and camera_idx in self.camera_motion:
            if self.motion_detected[camera_idx]:
                # Активный режим
                if current_time - self.last_motion_check.get(camera_idx, 0) > 0.5:
                    if self.prev_frames[camera_idx] is not None:
                        motion, contours = detect_motion(
                            self.prev_frames[camera_idx], frame,
                            self.MOTION_THRESHOLD, self.MOTION_MIN_AREA, None
                        )
                        if motion:
                            self.last_motion_time[camera_idx] = current_time
                    self.last_motion_check[camera_idx] = current_time
                    self.prev_frames[camera_idx] = frame.copy()

                # Проверка таймаута
                time_since_last_motion = current_time - self.last_motion_time[camera_idx]
                time_left = int(self.MOTION_TIMEOUT - time_since_last_motion)
                if time_since_last_motion > self.MOTION_TIMEOUT:
                    if camera_idx in self.active_motion_cameras:
                        duration = current_time - self.motion_start_time[camera_idx]
                        motion_logger.log_motion_stopped(camera_idx, duration, 0)
                        self.active_motion_cameras.remove(camera_idx)
                    self.motion_detected[camera_idx] = False
                    self.last_check_time[camera_idx] = current_time
                    return get_waiting_frame(camera_idx)

                # Активный кадр
                display_frame = draw_motion_visualization(frame, [], camera_idx, None, time_left)
                return display_frame

            else:
                # Режим ожидания
                time_since_last_check = current_time - self.last_check_time[camera_idx]
                if time_since_last_check >= self.CHECK_INTERVAL:
                    if self.prev_frames[camera_idx] is not None:
                        motion, _ = detect_motion(
                            self.prev_frames[camera_idx], frame,
                            self.MOTION_THRESHOLD, self.MOTION_MIN_AREA, None
                        )
                        if motion:
                            self.motion_detected[camera_idx] = True
                            self.last_motion_time[camera_idx] = current_time
                            self.motion_start_time[camera_idx] = current_time
                            self.last_motion_check[camera_idx] = current_time
                            motion_logger.log_motion_detected(camera_idx, is_triggered=True)
                            self.active_motion_cameras.add(camera_idx)
                            self.prev_frames[camera_idx] = frame.copy()
                            return draw_motion_visualization(frame, [], camera_idx, None, self.MOTION_TIMEOUT)
                    self.last_check_time[camera_idx] = current_time
                    self.prev_frames[camera_idx] = frame.copy()

                next_check = int(self.CHECK_INTERVAL - (current_time - self.last_check_time[camera_idx]))
                return get_waiting_frame(camera_idx, max(0, next_check))

        return frame

    def get_grid_frame(self):
        """Получение текущего кадра сетки для отправки клиенту"""
        frames = []
        current_time = time.time()
        
        for idx, cap in enumerate(self.caps):
            camera_idx = self.camera_indices[idx]
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    processed_frame = self.process_camera_frame(camera_idx, frame, current_time)
                else:
                    processed_frame = get_no_signal_frame(camera_idx)
            else:
                processed_frame = get_no_signal_frame(camera_idx)
            
            small_frame = cv2.resize(processed_frame, (320, 240))
            frames.append(small_frame)

        # Создаем сетку 2x2
        while len(frames) < 4:
            frames.append(get_no_signal_frame(len(frames)))
        
        grid = create_video_grid(frames, (2, 2), (640, 480))
        return grid

    def cleanup(self):
        """Очистка ресурсов"""
        for cam_idx in list(self.active_motion_cameras):
            duration = time.time() - self.motion_start_time[cam_idx]
            motion_logger.log_motion_stopped(cam_idx, duration, 0)

        motion_logger.log_system_event("Завершение работы системы")
        release_cameras(self.caps)


class OctoServer:
    def __init__(self):
        self.system = HeadlessSurveillanceSystem()
        self.system.initialize()
        self.running = True
        self.current_grid = None
        self.frame_lock = threading.Lock()
        
        self.system_thread = threading.Thread(target=self.run_system_loop)
        self.system_thread.daemon = True
        self.system_thread.start()

    def run_system_loop(self):
        """Основной цикл системы"""
        try:
            while self.running:
                grid_frame = self.system.get_grid_frame()
                
                with self.frame_lock:
                    self.current_grid = grid_frame.copy()
                
                time.sleep(0.033)
                
        except Exception as e:
            print(f"[SYSTEM] Ошибка: {e}")

    def get_grid_frame(self):
        with self.frame_lock:
            if self.current_grid is not None:
                return self.current_grid.copy()
            else:
                return np.zeros((480, 640, 3), dtype=np.uint8)

    def video_stream(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((HOST, PORT_VIDEO))
        server_socket.listen(5)
        print(f"[SERVER] Видео-сервер слушает на {HOST}:{PORT_VIDEO}")

        while self.running:
            try:
                conn, addr = server_socket.accept()
                print(f"[SERVER] Видео-клиент подключен: {addr}")
                
                client_thread = threading.Thread(target=self.handle_video_client, args=(conn, addr))
                client_thread.daemon = True
                client_thread.start()
                
            except Exception as e:
                if self.running:
                    print(f"[SERVER] Ошибка: {e}")

    def handle_video_client(self, conn, addr):
        try:
            while self.running:
                grid_frame = self.get_grid_frame()
                
                if grid_frame is not None:
                    success, buffer = cv2.imencode('.jpg', grid_frame, [
                        int(cv2.IMWRITE_JPEG_QUALITY), 80
                    ])
                    
                    if success:
                        data = pickle.dumps(buffer, protocol=pickle.HIGHEST_PROTOCOL)
                        message_size = struct.pack(">L", len(data))
                        
                        try:
                            conn.sendall(message_size + data)
                        except (BrokenPipeError, ConnectionResetError):
                            break
                
                time.sleep(0.033)
                
        except Exception as e:
            print(f"[SERVER] Ошибка: {e}")
        finally:
            try:
                conn.close()
            except:
                pass

    def command_listener(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((HOST, PORT_CMD))
        server_socket.listen(5)
        print(f"[SERVER] Командный сервер слушает на {HOST}:{PORT_CMD}")

        while self.running:
            try:
                conn, addr = server_socket.accept()
                print(f"[SERVER] Командный клиент подключен: {addr}")
                
                client_thread = threading.Thread(target=self.handle_command_client, args=(conn, addr))
                client_thread.daemon = True
                client_thread.start()
                
            except Exception as e:
                if self.running:
                    print(f"[SERVER] Ошибка: {e}")

    def handle_command_client(self, conn, addr):
        try:
            while self.running:
                conn.settimeout(1.0)
                
                try:
                    data = conn.recv(4096)
                    if not data:
                        break
                except socket.timeout:
                    continue
                
                try:
                    cmd = json.loads(data.decode("utf-8"))
                    print(f"[SERVER] Команда от {addr}: {cmd}")

                    response = {"status": "ok", "command": cmd["action"]}
                    
                    if cmd["action"] == "set_timeout":
                        self.system.MOTION_TIMEOUT = int(cmd["value"])
                    elif cmd["action"] == "set_threshold":
                        self.system.MOTION_THRESHOLD = int(cmd["value"])
                    elif cmd["action"] == "enable_face":
                        cam = cmd["camera"]
                        if cam not in self.system.camera_faces:
                            self.system.camera_faces.append(cam)
                    elif cmd["action"] == "disable_face":
                        cam = cmd["camera"]
                        if cam in self.system.camera_faces:
                            self.system.camera_faces.remove(cam)
                    elif cmd["action"] == "enable_motion":
                        cam = cmd["camera"]
                        if cam not in self.system.camera_motion:
                            self.system.camera_motion.append(cam)
                    elif cmd["action"] == "disable_motion":
                        cam = cmd["camera"]
                        if cam in self.system.camera_motion:
                            self.system.camera_motion.remove(cam)
                    elif cmd["action"] == "quit":
                        self.running = False
                    
                    conn.send(json.dumps(response).encode("utf-8"))

                except Exception as e:
                    response = {"status": "error", "message": str(e)}
                    conn.send(json.dumps(response).encode("utf-8"))
                    
        except Exception as e:
            print(f"[SERVER] Ошибка: {e}")
        finally:
            try:
                conn.close()
            except:
                pass

    def stop(self):
        self.running = False
        self.system.cleanup()

    def run(self):
        try:
            print("[SERVER] Запуск сервера...")
            
            video_thread = threading.Thread(target=self.video_stream)
            command_thread = threading.Thread(target=self.command_listener)
            
            video_thread.daemon = True
            command_thread.daemon = True
            
            video_thread.start()
            command_thread.start()
            
            print("[SERVER] Сервер запущен!")
            print("[SERVER] Для остановки: Ctrl+C")
            
            while self.running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n[SERVER] Остановка...")
        finally:
            self.stop()


if __name__ == "__main__":
    server = OctoServer()
    server.run()\
