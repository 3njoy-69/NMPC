import socket
import numpy as np
import json
from scipy.interpolate import CubicSpline
import casadi as ca

# ==== CÁC THÔNG SỐ CƠ BẢN ====
L = 43  # chiều dài cơ sở
dt = 0.1  # bước thời gian
N = 30  # Horizon

v_max, v_min = 2.0, -2.0
delta_max = np.deg2rad(30)
a_max = 1
omega_max = np.deg2rad(25)


# ==== CÁC HÀM ====
# Mô hình xe
def vehicle_model(x, u):
    px, py, yaw = x[0], x[1], x[2]
    v, delta = u[0], u[1]
    px_next = px + v * np.cos(yaw) * dt
    py_next = py + v * np.sin(yaw) * dt
    yaw_next = yaw + v / L * np.tan(delta) * dt
    return np.array([px_next, py_next, yaw_next])


# Làm trơn quỹ đạo
def smooth_path(x, y):
    t = np.linspace(0, 1, len(x))
    cs_x = CubicSpline(t, x)
    cs_y = CubicSpline(t, y)
    t_new = np.linspace(0, 1, 500)
    x_smooth = cs_x(t_new)
    y_smooth = cs_y(t_new)
    yaw_smooth = np.arctan2(np.gradient(y_smooth), np.gradient(x_smooth))
    return x_smooth, y_smooth, yaw_smooth


# NMPC
def solve_nmpc(state, ref_traj):
    opti = ca.Opti()

    X = opti.variable(3, N + 1)  # state: x, y, yaw
    U = opti.variable(2, N)  # control: v, delta

    x0 = opti.parameter(3)
    x_ref = opti.parameter(3, N + 1)

    Q = ca.diagcat(50, 50, 1)
    R = ca.diagcat(0.1, 0.1)

    opti.subject_to(X[:, 0] == x0)

    for k in range(N):
        x_next = X[0, k] + U[0, k] * ca.cos(X[2, k]) * dt
        y_next = X[1, k] + U[0, k] * ca.sin(X[2, k]) * dt
        yaw_next = X[2, k] + U[0, k] / L * ca.tan(U[1, k]) * dt
        opti.subject_to(X[:, k + 1] == ca.vertcat(x_next, y_next, yaw_next))

    cost = 0
    for k in range(N):
        cost += ca.mtimes([(X[:, k] - x_ref[:, k]).T, Q, (X[:, k] - x_ref[:, k])])
        cost += ca.mtimes([U[:, k].T, R, U[:, k]])

    opti.minimize(cost)

    opti.subject_to(opti.bounded(v_min, U[0, :], v_max))
    opti.subject_to(opti.bounded(-delta_max, U[1, :], delta_max))

    opti.solver("ipopt")
    opti.set_value(x0, state)
    opti.set_value(x_ref, ref_traj)

    try:
        sol = opti.solve()
        u0 = sol.value(U[:, 0])
        return u0
    except:
        print("NMPC failed")
        return np.array([0.0, 0.0])


# Gửi điều khiển qua TCP
import socket

def send_control_tcp(v, delta, ip, port):
    # Tạo kết nối socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((ip, port))
            # Chuyển đổi điều khiển thành chuỗi byte và gửi
            control_data = f"{v},{delta}\n"
            s.sendall(control_data.encode())
        except ConnectionRefusedError as e:
            print(f"Không thể kết nối đến {ip}:{port}. Lỗi: {e}")



# Chạy mô phỏng
def run_simulation(x_ref, y_ref, yaw_ref, ip, port):
    # Thêm các điểm cuối để giữ đủ chiều dài cho NMPC
    for _ in range(N):
        x_ref = np.append(x_ref, x_ref[-1])
        y_ref = np.append(y_ref, y_ref[-1])
        yaw_ref = np.append(yaw_ref, yaw_ref[-1])

    traj = []
    x = np.array([x_ref[0], y_ref[0], yaw_ref[0]])

    total_steps = len(x_ref)
    final_target = np.array([x_ref[-1], y_ref[-1]])

    for t in range(total_steps - N):
        ref = np.vstack([x_ref[t:t + N + 1], y_ref[t:t + N + 1], yaw_ref[t:t + N + 1]])
        u = solve_nmpc(x, ref)

        # Gửi điều khiển qua TCP
        send_control_tcp(u[0], u[1], ip, port)  # Gửi điều khiển mỗi bước
        print(f"Bước {t}: u = [v: {u[0]:.2f}, delta: {np.rad2deg(u[1]):.2f}]")

        dist_to_goal = np.linalg.norm(x[:2] - final_target)

        # Nếu gần đích rồi thì dừng sớm
        if dist_to_goal < 0.3:
            print(f"✅ Đã đến rất gần đích tại bước {t}, vị trí: {x[:2]}")
            break

        # Nếu u gần như 0 và còn xa đích thì có thể bị kẹt
        if np.linalg.norm(u) < 1e-2 and dist_to_goal > 0.5:
            print(f"⚠️  NMPC output nhỏ tại bước {t}, vị trí: {x[:2]} → dừng.")
            break

        x = vehicle_model(x, u)
        traj.append(x)

    # Ép xe chạy thêm 2 bước về đích nếu còn xa
    while np.linalg.norm(x[:2] - final_target) > 0.3:
        print(f"➡️  Ép bám đích, vị trí hiện tại: {x[:2]}")
        ref = np.tile([[x_ref[-1]], [y_ref[-1]], [yaw_ref[-1]]], (1, N + 1))
        u = solve_nmpc(x, ref)
        send_control_tcp(u[0], u[1], ip, port)  # Gửi điều khiển mỗi bước
        x = vehicle_model(x, u)
        traj.append(x)

    return np.array(traj)


# ==== MAIN ====
ip = "127.0.0.1"  # Địa chỉ IP của máy chủ nhận dữ liệu
port = 5000  # Cổng TCP

with open('path_data.json', 'r') as f:
    path = json.load(f)
x_raw = [p[0] for p in path]
y_raw = [p[1] for p in path]
yaw_raw = [p[2] for p in path]

x_ref, y_ref, yaw_ref = smooth_path(x_raw, y_raw)
states = run_simulation(x_ref, y_ref, yaw_ref, ip, port)


