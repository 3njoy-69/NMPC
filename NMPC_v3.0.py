import socket
import numpy as np
import json
from scipy.interpolate import CubicSpline
import casadi as ca
import time

# ==== THÔNG SỐ CƠ BẢN ====
L = 43
dt = 0.1
N = 30
v_max, v_min = 2.0, -2.0
delta_max = np.deg2rad(30)


# ==== MÔ HÌNH XE ====
def vehicle_model(x, u):
    px, py, yaw = x[0], x[1], x[2]
    v, delta = u[0], u[1]
    px_next = px + v * np.cos(yaw) * dt
    py_next = py + v * np.sin(yaw) * dt
    yaw_next = yaw + v / L * np.tan(delta) * dt
    return np.array([px_next, py_next, yaw_next])


# ==== LÀM TRƠN QUỸ ĐẠO ====
def smooth_path(x, y):
    t = np.linspace(0, 1, len(x))
    cs_x = CubicSpline(t, x)
    cs_y = CubicSpline(t, y)
    t_new = np.linspace(0, 1, 500)
    x_smooth = cs_x(t_new)
    y_smooth = cs_y(t_new)
    yaw_smooth = np.arctan2(np.gradient(y_smooth), np.gradient(x_smooth))
    return x_smooth, y_smooth, yaw_smooth


# ==== GIẢI NMPC ====
def solve_nmpc(state, ref_traj):
    opti = ca.Opti()

    X = opti.variable(3, N + 1)  # state
    U = opti.variable(2, N)      # control

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
        return sol.value(U[:, 0])
    except:
        print("❌ NMPC không hội tụ.")
        return np.array([0.0, 0.0])


# ==== GỬI LỆNH ĐIỀU KHIỂN ====
def send_control_tcp(sock, v, delta):
    try:
        control_data = f"{v},{delta}\n"
        sock.sendall(control_data.encode())
    except Exception as e:
        print(f"❌ Lỗi gửi TCP: {e}")


# ==== MÔ PHỎNG ====
def run_simulation(x_ref, y_ref, yaw_ref, ip, port):
    # Thêm điểm cuối để đủ độ dài horizon
    for _ in range(N):
        x_ref = np.append(x_ref, x_ref[-1])
        y_ref = np.append(y_ref, y_ref[-1])
        yaw_ref = np.append(yaw_ref, yaw_ref[-1])

    x = np.array([x_ref[0], y_ref[0], yaw_ref[0]])
    final_target = np.array([x_ref[-1], y_ref[-1]])
    traj = []

    # Mở kết nối socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((ip, port))
        print("✅ Đã kết nối đến server")
    except Exception as e:
        print(f"❌ Không kết nối được TCP: {e}")
        return []

    for t in range(len(x_ref) - N):
        ref = np.vstack([x_ref[t:t + N + 1], y_ref[t:t + N + 1], yaw_ref[t:t + N + 1]])
        u = solve_nmpc(x, ref)

        send_control_tcp(sock, u[0], u[1])
        print(f"🔧 Bước {t}: v = {u[0]:.2f}, δ = {np.rad2deg(u[1]):.2f}°")

        dist = np.linalg.norm(x[:2] - final_target)
        if dist < 0.3:
            print(f"🏁 Đã đến đích ở bước {t}")
            break

        x = vehicle_model(x, u)
        traj.append(x)

    sock.close()
    print("🔌 Đã đóng kết nối TCP.")
    return np.array(traj)


# ==== MAIN ====
if __name__ == "__main__":
    ip = "127.0.0.1"
    port = 5000

    with open('path_data.json', 'r') as f:
        path = json.load(f)
    x_raw = [p[0] for p in path]
    y_raw = [p[1] for p in path]
    yaw_raw = [p[2] for p in path]

    x_ref, y_ref, yaw_ref = smooth_path(x_raw, y_raw)
    run_simulation(x_ref, y_ref, yaw_ref, ip, port)
