import cv2
import numpy as np
import os

def initialize_cameras(camera_indices):
    """Инициализация камер"""
    caps = []
    for idx in camera_indices:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            print(f"\033[32mУдалось инициализирована камеру {idx}\033[0m")
        else:
            print(f"\033[91mНе удалось инициализировать камеру {idx}\033[0m")
        caps.append(cap)
    return caps

def release_cameras(caps):
    """Освобождение ресурсов камер"""
    for cap in caps:
        if cap.isOpened():
            cap.release()
    print("Ресурсы камер освобождены")

def create_video_grid(frames, grid_size=(2, 2), output_size=(640, 480)):
    """Создание сетки из кадров"""
    if not frames:
        return np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
    
    resized_frames = [cv2.resize(frame, (output_size[0] // grid_size[1], 
                                       output_size[1] // grid_size[0])) 
                     for frame in frames]
    
    rows = []
    for i in range(0, len(resized_frames), grid_size[1]):
        row_frames = resized_frames[i:i + grid_size[1]]
        if len(row_frames) < grid_size[1]:
            empty_frames = [np.zeros_like(row_frames[0]) for _ in range(grid_size[1] - len(row_frames))]
            row_frames.extend(empty_frames)
        row = np.hstack(row_frames)
        rows.append(row)
    
    grid = np.vstack(rows[:grid_size[0]])
    return grid

def get_no_signal_frame(camera_idx, size=(640, 480)):
    """Создание кадра 'Нет сигнала'"""
    frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    cv2.putText(frame, f"No signal {camera_idx}", 
               (240, size[1] // 2), cv2.FONT_HERSHEY_SIMPLEX, 
               1, (0, 0, 255), 2)
    return frame

def get_waiting_frame(camera_idx, time_left=None, size=(640, 480)):
    """Создание кадра 'Ожидание движения'"""
    frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    cv2.putText(frame, f"CAMERA {camera_idx}", 
               (240, size[1] // 2 - 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, "WAITING FOR MOTION", 
               (size[0] // 2 - 120, size[1] // 2), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if time_left is not None:
        cv2.putText(frame, f"Next check: {time_left}s", 
                   (240, size[1] // 2 + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, "Standby mode", 
               (240, size[1] - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return frame

class MaskCreator:
    def create_mask(self, camera_index, mask_name="default"):
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"\033[91mНе удалось открыть камеру {camera_index}\033[0m")
            return None

        print(f"Создание маски для камеры {camera_index}")
        print("Инструкция:")
        print("1. 's' - начать/закончить рисование")
        print("2. ЛКМ - добавить точку многоугольника")
        print("3. ПКМ - удалить последнюю точку")
        print("4. 'c' - очистить все точки")
        print("5. 'q' - сохранить и выйти")
        print("6. ESC - выйти без сохранения")

        os.makedirs("masks", exist_ok=True)
        mask_path = f"masks/camera_{camera_index}_{mask_name}.png"

        points = []
        drawing = False

        def mouse_callback(event, x, y,flags, param):
            nonlocal points, drawing
            if event == cv2.EVENT_LBUTTONDOWN and drawing:
                points.append((x, y))
            elif event == cv2.EVENT_RBUTTONDOWN and drawing and points:
                points.pop()

        cv2.namedWindow("Create Mask")
        cv2.setMouseCallback("Create Mask", mouse_callback)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("\033[91mНе удалось получить кадр\033[0m")
                break

            display_frame = frame.copy()

            # Рисуем линии и точки поверх текущего кадра
            if len(points) > 1:
                for i in range(len(points)-1):
                    cv2.line(display_frame, points[i], points[i+1], (0,255,0),2)
                if drawing:
                    cv2.line(display_frame, points[-1], cv2.getWindowImageRect("Create Mask")[:2], (0,255,255),1)
            for pt in points:
                cv2.circle(display_frame, pt, 5, (0,0,255), -1)

            cv2.imshow("Create Mask", display_frame)
            key = cv2.waitKey(30) & 0xFF

            if key == ord('s'):
                drawing = not drawing
                print(f"Рисование {'начато' if drawing else 'закончено'}")
            elif key == ord('c'):
                points = []
                print("Очистка точек")
            elif key == ord('q'):
                if len(points) >= 3:
                    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    pts = np.array(points, np.int32)
                    cv2.fillPoly(mask, [pts], 255)
                    cv2.imwrite(mask_path, mask)
                    print(f"Маска сохранена: {mask_path}")
                    cap.release()
                    cv2.destroyAllWindows()
                    return mask_path
                else:
                    print("\033[93mНужно >= 3 точки\033[0m")
            elif key == 27:
                print("\033[93mВыход без сохранения\033[0m")
                cap.release()
                cv2.destroyAllWindows()
                return None

def load_mask(mask_path):
    """Загружает маску из файла"""
    if os.path.exists(mask_path):
        return cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return None

def overlay_mask(frame, mask, color=(0,255,0), alpha=0.3):
    """Накладывает прозрачную маску на кадр"""
    if mask is None:
        return frame
    overlay = frame.copy()
    mask_3ch = cv2.merge([mask, mask, mask])
    color_layer = np.zeros_like(frame)
    color_layer[:] = color
    overlay = np.where(mask_3ch != 0, cv2.addWeighted(color_layer, alpha, overlay, 1 - alpha, 0), overlay)
    return overlay

def draw_bounding_box(frame, rect, label=None, color=(0, 255, 0)):
    """Унифицированное рисование bounding box"""
    x, y, w, h = rect
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    if label:
        cv2.putText(frame, label, (x + 5, y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
