import socket
import struct
import pickle
import cv2
import json
import threading
import time

def discover_server():
    possible_ips = ["192.168.1.100", "192.168.0.100"]
    
    for ip in possible_ips:
        try:
            print(f"Проверка {ip}...")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex((ip, 9999))
            sock.close()
            
            if result == 0:
                print(f"Сервер найден: {ip}")
                return ip
        except:
            continue
    
    return None

HOST = discover_server()

if HOST is None:
    HOST = input("Введите IP адрес Raspberry Pi: ")

PORT_VIDEO = 9999
PORT_CMD = 9998

print(f"Подключение к {HOST}")

def video_receiver():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    try:
        print(f"Подключаюсь к {HOST}:{PORT_VIDEO}...")
        client_socket.connect((HOST, PORT_VIDEO))
        client_socket.settimeout(5.0)
        print("Подключено к видео-серверу!")
    except Exception as e:
        print(f"Ошибка подключения: {e}")
        return

    data = b""
    payload_size = struct.calcsize(">L")

    try:
        while True:
            while len(data) < payload_size:
                try:
                    packet = client_socket.recv(4096)
                    if not packet:
                        return
                    data += packet
                except socket.timeout:
                    continue

            packed_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack(">L", packed_size)[0]

            while len(data) < msg_size:
                try:
                    packet = client_socket.recv(4096)
                    if not packet:
                        return
                    data += packet
                except socket.timeout:
                    continue

            frame_data = data[:msg_size]
            data = data[msg_size:]

            try:
                buffer = pickle.loads(frame_data)
                frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

                if frame is not None:
                    cv2.imshow("Raspberry Pi Surveillance", frame)
                else:
                    print("Ошибка декодирования кадра")
                    
            except Exception as e:
                print(f"Ошибка кадра: {e}")
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Ошибка видео: {e}")
    finally:
        client_socket.close()
        cv2.destroyAllWindows()

def send_command(cmd):
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.settimeout(5)
        client_socket.connect((HOST, PORT_CMD))
        client_socket.send(json.dumps(cmd).encode("utf-8"))
        
        response = client_socket.recv(1024)
        if response:
            print(f"Ответ: {json.loads(response.decode('utf-8'))}")
                
    except Exception as e:
        print(f"Ошибка команды: {e}")
    finally:
        try:
            client_socket.close()
        except:
            pass

if __name__ == "__main__":
    print("Клиент системы видеонаблюдения")
    
    t1 = threading.Thread(target=video_receiver)
    t1.daemon = True
    t1.start()

    time.sleep(2)

    while True:
        print("\nМеню:")
        print("1. Установить таймаут")
        print("2. Установить чувствительность")
        print("3. Включить детектор лиц (Cam0)")
        print("4. Выключить детектор лиц (Cam0)")
        print("5. Включить детектор движения (Cam0)")
        print("6. Выключить детектор движения (Cam0)")
        print("q. Выйти")
        
        choice = input("➡ ").strip()

        if choice == "1":
            val = input("Таймаут (сек): ")
            send_command({"action": "set_timeout", "value": val})
        elif choice == "2":
            val = input("Чувствительность: ")
            send_command({"action": "set_threshold", "value": val})
        elif choice == "3":
            send_command({"action": "enable_face", "camera": 0})
        elif choice == "4":
            send_command({"action": "disable_face", "camera": 0})
        elif choice == "5":
            send_command({"action": "enable_motion", "camera": 0})
        elif choice == "6":
            send_command({"action": "disable_motion", "camera": 0})
        elif choice == "q":
            break

    print("Выход")