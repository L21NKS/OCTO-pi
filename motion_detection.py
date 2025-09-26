import cv2
import numpy as np
import datetime
from camera_utils import overlay_mask, get_no_signal_frame, draw_bounding_box

def detect_motion(prev_frame, current_frame, threshold=25, min_area=500, mask=None):
    """
    Детектирование движения между двумя кадрами с улучшенным трекингом
    """
    if prev_frame is None or current_frame is None:
        return False, []
    
    # Конвертация в оттенки серого
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    
    # Размытие для уменьшения шума
    prev_blur = cv2.GaussianBlur(prev_gray, (21, 21), 0)
    current_blur = cv2.GaussianBlur(current_gray, (21, 21), 0)
    
    # Применение маски если она есть
    if mask is not None:
        prev_blur = cv2.bitwise_and(prev_blur, prev_blur, mask=cv2.bitwise_not(mask))
        current_blur = cv2.bitwise_and(current_blur, current_blur, mask=cv2.bitwise_not(mask))
    
    # Вычисление разницы между кадрами
    frame_delta = cv2.absdiff(prev_blur, current_blur)
    
    # Бинаризация разницы
    thresh = cv2.threshold(frame_delta, threshold, 255, cv2.THRESH_BINARY)[1]
    
    # Морфологические операции для улучшения детектирования
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    # Поиск контуров
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    motion_detected = False
    significant_contours = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            motion_detected = True
            significant_contours.append(contour)
    
    return motion_detected, significant_contours

def draw_motion_visualization(frame, contours, camera_idx, mask=None, time_left=None):
    """
    Отрисовка визуализации движения на кадре с нумерацией объектов
    """
    if frame is None:
        return get_no_signal_frame(camera_idx)
    
    output_frame = frame.copy()
    
    # Накладываем маску если есть
    if mask is not None:
        output_frame = overlay_mask(output_frame, mask)
    
    object_count = 0
    
    # Рисуем bounding boxes и контуры для каждого движения
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        draw_bounding_box(output_frame, (x, y, w, h), label=str(i + 1), color=(0, 0, 255))
        
        # Контур
        cv2.drawContours(output_frame, [contour], -1, (0, 255, 255), 1)
        
        # Размер области движения
        cv2.putText(output_frame, f"{w}x{h}", (x, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        object_count += 1
    
    # Добавляем текст с информацией
    if contours:
        cv2.putText(output_frame, f"Motion: {len(contours)} objects", 
                   (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Временная метка
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        cv2.putText(output_frame, timestamp, (10, output_frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Номер камеры
        cv2.putText(output_frame, f"Cam {camera_idx}", 
                   (130, output_frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Индикатор маски
        if mask is not None:
            cv2.putText(output_frame, "Mask active", 
                       (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:
        cv2.putText(output_frame, "No motion", 
                   (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return output_frame
