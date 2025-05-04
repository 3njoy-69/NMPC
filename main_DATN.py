import numpy as np
import matplotlib.pyplot as plt
import socket
from hybridAStar_v4 import run
from hybridAStar_v4 import calculateMapParameters
from hybridAStar_v4 import map
from NMPC_v6 import control


def autoParking():
    # --- SET UP THÔNG SỐ TÌM ĐƯỜNG VÀ GỌI FILE hybridAStar ---
    s = [160, 150, np.deg2rad(0)]
    g = [250, 80, np.deg2rad(90)]

    obstacleX, obstacleY = map()

    mapParameters = calculateMapParameters(obstacleX, obstacleY, 4, np.deg2rad(15.0))

    run(s, g, mapParameters, plt)

    control()

# Tạo socket để nhận dữ liệu từ Qt (trên cổng 12347)
udp_receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_receiver.bind(("127.0.0.1", 12347))
print("Python đang lắng nghe trên cổng 12347 để nhận dữ liệu từ Qt...")

# Tạo socket riêng để gửi tín hiệu đến Qt (trên cổng 12346)
udp_sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
qt_address = ("127.0.0.1", 12346)

# Gửi tín hiệu ban đầu đến Qt
udp_sender.sendto("parking signal".encode(), qt_address)

def main():
    # while True:
    #     data, addr = udp_receiver.recvfrom(1024)
    #     message2 = data.decode()
    #     print(f"Nhận từ {addr}: {message2}")
    #     if message2 == "Yes":
    #         print("🚗 Bật chế độ đỗ xe tự động!")
    #         autoParking()
    #         break
    #     elif message2 == "No":
    #         print("⛔ Không kích hoạt chế độ đỗ xe tự động!")
    #         break

    autoParking()

if __name__ == "__main__":
    main()



