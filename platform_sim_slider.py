import pygame
import math
import json
import sys
from pathlib import Path
from collections import deque
import matplotlib.pyplot as plt


CONFIG_PATH = Path("config_stewart.json")

try:
    with open(CONFIG_PATH, "r") as f:
        cfg = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError(f"Configuration file not found at {CONFIG_PATH}")  

# Extract parameters
platform_type = cfg.get("platform_type", "circular")
assert platform_type == "circular", "This demo assumes a circular platform."

platform_radius_m = cfg["platform_radius_m"]
platform_center_px = cfg["platform_center_pixels"]
platform_radius_px = cfg["platform_radius_pixels"]

cam_w = cfg["camera"]["frame_width"]
cam_h = cfg["camera"]["frame_height"]

calib = cfg["calibration"]
px2m_x = calib["pixel_to_meter_ratio_x"]
px2m_y = calib["pixel_to_meter_ratio_y"]

servo_cfg = cfg["servo"]
neutral_angles = servo_cfg.get("neutral_angles", [15.0, 15.0, 15.0])
motor_dir_invert = servo_cfg.get("motor_direction_invert", [False, False, False])

pid_cfg = cfg["pid"]
# Initial PID values from config
INIT_KP = pid_cfg["Kp_x"]
INIT_KI = pid_cfg["Ki_x"]
INIT_KD = pid_cfg["Kd_x"]

platform_cfg = cfg["platform"]
max_roll_deg = platform_cfg["max_roll_angle"]
max_pitch_deg = platform_cfg["max_pitch_angle"]
roll_dir_invert = False  

motor_pos_px = cfg["motor_positions_pixels"]
motor_angles_deg = cfg["motor_angles_deg"]
axis_rotation_deg = cfg["axis_rotation_deg"]

# Constants
g = 9.81  
max_roll_rad = math.radians(max_roll_deg)
max_pitch_rad = math.radians(max_pitch_deg)

WORKSPACE_RADIUS_M = platform_radius_m * 0.95

height_for_max_angle = platform_radius_m * math.tan(max_roll_rad if max_roll_rad != 0 else 1e-6)
if abs(height_for_max_angle) < 1e-9:
    height_for_max_angle = 1e-6
servo_deg_per_m = max_roll_deg / height_for_max_angle

# helper func 
def clamp(v, vmin, vmax):
    return max(vmin, min(vmax, v))

def px_to_m(px, py):
    cx, cy = platform_center_px
    dx = (px - cx) * px2m_x
    dy = (cy - py) * px2m_y  # invert Y so up is +y
    return dx, dy

def m_to_px(x, y):
    cx, cy = platform_center_px
    px = cx + x / px2m_x
    py = cy - y / px2m_y
    return int(px), int(py)

def clamp_to_workspace(x, y):
    r = math.hypot(x, y)
    if r > WORKSPACE_RADIUS_M:
        scale = WORKSPACE_RADIUS_M / r
        x *= scale
        y *= scale
    return x, y

def compute_servo_angles(roll_rad, pitch_rad):
    angles = []
    for (mx, my), neutral, invert in zip(motor_pos_px, neutral_angles, motor_dir_invert):
        x_m, y_m = px_to_m(mx, my)
        z = x_m * math.sin(roll_rad) + y_m * math.sin(pitch_rad)
        d_angle = servo_deg_per_m * z
        if invert:
            d_angle = -d_angle
        angles.append(neutral + d_angle)
    return angles

# slider for PID tuning 
class Slider:
    def __init__(self, x, y, w, h, min_val, max_val, init_val, label, font):
        self.rect = pygame.Rect(x, y, w, h)
        self.min_val = min_val
        self.max_val = max_val
        self.value = clamp(init_val, min_val, max_val)
        self.label = label
        self.font = font
        self.dragging = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.dragging = True
                self._update_from_mouse(event.pos[0])
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION:
            if self.dragging:
                self._update_from_mouse(event.pos[0])

    def _update_from_mouse(self, mouse_x):
        x = clamp(mouse_x, self.rect.left, self.rect.right)
        t = (x - self.rect.left) / self.rect.width
        self.value = self.min_val + t * (self.max_val - self.min_val)

    def draw(self, surface):
        pygame.draw.rect(surface, (120, 120, 120), self.rect, border_radius=4)
        t = (self.value - self.min_val) / (self.max_val - self.min_val) if self.max_val > self.min_val else 0.0
        knob_x = self.rect.left + t * self.rect.width
        knob_y = self.rect.centery
        pygame.draw.circle(surface, (255, 255, 255), (int(knob_x), int(knob_y)), self.rect.height // 2)

        label_surf = self.font.render(
            f"{self.label}: {self.value:.3f}", True, (230, 230, 230)
        )
        surface.blit(label_surf, (self.rect.left, self.rect.top - 18))

# pygame 
pygame.init()
screen = pygame.display.set_mode((cam_w, cam_h))
pygame.display.set_caption("2D Stewart Platform Ball Balancing - Simulation")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 16)

# Sliders at bottom
slider_width = cam_w - 40
slider_height = 10
ui_panel_height = 100          
base_y = cam_h 
spacing = 20  

kp_slider = Slider(
    x=20,
    y=base_y - 3 * spacing,
    w=slider_width,
    h=slider_height,
    min_val=0.0,
    max_val=3.0,
    init_val=INIT_KP,
    label="Kp",
    font=font,
)

ki_slider = Slider(
    x=20,
    y=base_y - 2 * spacing,
    w=slider_width,
    h=slider_height,
    min_val=0.0,
    max_val=0.3,
    init_val=INIT_KI,
    label="Ki",
    font=font,
)

kd_slider = Slider(
    x=20,
    y=base_y - spacing,
    w=slider_width,
    h=slider_height,
    min_val=0.0,
    max_val=5.0,
    init_val=INIT_KD,
    label="Kd",
    font=font,
)


ball_pos = [0.0, 0.0]  
ball_vel = [0.0, 0.0]  
target_pos = [0.0, 0.0]

# PID state
err_int_x = 0.0
err_prev_x = 0.0
err_int_y = 0.0
err_prev_y = 0.0

current_roll_cmd = 0.0
current_pitch_cmd = 0.0

# Logging for plots
log_t = []
log_ball_x = []
log_ball_y = []
log_roll = []
log_pitch = []
log_servo1 = []
log_servo2 = []
log_servo3 = []

traj_history = deque(maxlen=2000)

sim_running = False
time_s = 0.0

def reset_sim_to_click(px, py):
    global ball_pos, ball_vel, err_int_x, err_int_y, err_prev_x, err_prev_y
    global traj_history, sim_running, time_s
    global log_t, log_ball_x, log_ball_y, log_roll, log_pitch, log_servo1, log_servo2, log_servo3
    global current_roll_cmd, current_pitch_cmd

    x, y = px_to_m(px, py)
    x, y = clamp_to_workspace(x, y)

    ball_pos = [x, y]
    ball_vel = [0.0, 0.0]

    err_int_x = err_int_y = 0.0
    err_prev_x = err_prev_y = 0.0

    current_roll_cmd = 0.0
    current_pitch_cmd = 0.0

    traj_history.clear()
    traj_history.append((x, y))

    sim_running = True
    time_s = 0.0

    log_t.clear()
    log_ball_x.clear()
    log_ball_y.clear()
    log_roll.clear()
    log_pitch.clear()
    log_servo1.clear()
    log_servo2.clear()
    log_servo3.clear()

def step_sim(dt, Kp, Ki, Kd):
    global ball_pos, ball_vel, err_int_x, err_int_y, err_prev_x, err_prev_y, time_s
    global current_roll_cmd, current_pitch_cmd

    # PID control
    # x-axis
    err_x = target_pos[0] - ball_pos[0]
    err_int_x += err_x * dt
    err_der_x = (err_x - err_prev_x) / dt
    err_prev_x = err_x

    roll_cmd = Kp * err_x + Ki * err_int_x + Kd * err_der_x
    if roll_dir_invert:
        roll_cmd = -roll_cmd
    roll_cmd = clamp(roll_cmd, -max_roll_rad, max_roll_rad)

    # y-axis
    err_y = target_pos[1] - ball_pos[1]
    err_int_y += err_y * dt
    err_der_y = (err_y - err_prev_y) / dt
    err_prev_y = err_y

    pitch_cmd = Kp * err_y + Ki * err_int_y + Kd * err_der_y
    pitch_cmd = clamp(pitch_cmd, -max_pitch_rad, max_pitch_rad)

    current_roll_cmd = roll_cmd
    current_pitch_cmd = pitch_cmd

    # BALL DYNAMICS
    vel_damping = 3.0  # tweak if needed
    ax = g * math.sin(roll_cmd) - vel_damping * ball_vel[0]
    ay = g * math.sin(pitch_cmd) - vel_damping * ball_vel[1]

    ball_vel[0] += ax * dt
    ball_vel[1] += ay * dt

    ball_pos[0] += ball_vel[0] * dt
    ball_pos[1] += ball_vel[1] * dt

    # clamp to circular workspace
    ball_pos[0], ball_pos[1] = clamp_to_workspace(ball_pos[0], ball_pos[1])

    traj_history.append(tuple(ball_pos))
    time_s += dt

    # Logging
    s1, s2, s3 = compute_servo_angles(roll_cmd, pitch_cmd)
    log_t.append(time_s)
    log_ball_x.append(ball_pos[0])
    log_ball_y.append(ball_pos[1])
    log_roll.append(roll_cmd)
    log_pitch.append(pitch_cmd)
    log_servo1.append(s1)
    log_servo2.append(s2)
    log_servo3.append(s3)

def draw():
    screen.fill((30, 30, 30))

    # platform
    pygame.draw.circle(
        screen,
        (80, 80, 80),
        platform_center_px,
        int(platform_radius_px),
        width=2,
    )

    # visual tilt indicator
    cx, cy = platform_center_px
    tilt_scale = platform_radius_px * 0.6
    dx = tilt_scale * math.sin(current_roll_cmd)
    dy = tilt_scale * math.sin(current_pitch_cmd)

    pygame.draw.line(
        screen,
        (255, 100, 100),
        (cx - dx, cy + dy),
        (cx + dx, cy - dy),
        4,
    )

    # motor positions
    for (mx, my) in motor_pos_px:
        pygame.draw.circle(screen, (0, 150, 255), (mx, my), 6)

    # trajectory
    if len(traj_history) > 1:
        pts = [m_to_px(x, y) for (x, y) in traj_history]
        pygame.draw.lines(screen, (0, 255, 0), False, pts, 2)

    # ball
    bx, by = m_to_px(ball_pos[0], ball_pos[1])
    pygame.draw.circle(screen, (255, 200, 0), (bx, by), 8)

    # crosshair at center
    pygame.draw.line(screen, (120, 120, 120), (cx - 10, cy), (cx + 10, cy), 1)
    pygame.draw.line(screen, (120, 120, 120), (cx, cy - 10), (cx, cy + 10), 1)

    # Instructions
    txt1 = font.render(
        "Click inside circle to place ball. ESC to quit.",
        True, (200, 200, 200)
    )
    txt2 = font.render(
        f"Ball pos (m): x={ball_pos[0]:+.3f}, y={ball_pos[1]:+.3f}",
        True, (200, 200, 200)
    )
    screen.blit(txt1, (10, 10))
    screen.blit(txt2, (10, 30))

    # Draw sliders
    kp_slider.draw(screen)
    ki_slider.draw(screen)
    kd_slider.draw(screen)

    pygame.display.flip()

def show_plots():
    if not log_t:
        return
    import numpy as np

    t = np.array(log_t)
    bx = np.array(log_ball_x)
    by = np.array(log_ball_y)
    roll_deg = np.degrees(np.array(log_roll))
    pitch_deg = np.degrees(np.array(log_pitch))
    s1 = np.array(log_servo1)
    s2 = np.array(log_servo2)
    s3 = np.array(log_servo3)

    # Ball trajectory vs time
    plt.figure()
    plt.title("Ball position vs time")
    plt.plot(t, bx, label="x (m)")
    plt.plot(t, by, label="y (m)")
    plt.xlabel("time (s)")
    plt.ylabel("position (m)")
    plt.legend()
    plt.grid(True)

    # Platform angles vs time (deg)
    plt.figure()
    plt.title("Platform angles vs time (degrees)")
    plt.plot(t, roll_deg, label="roll (deg)")
    plt.plot(t, pitch_deg, label="pitch (deg)")
    plt.xlabel("time (s)")
    plt.ylabel("angle (deg)")
    plt.legend()
    plt.grid(True)

    # All motor servo angles vs time (deg)
    plt.figure()
    plt.title("Servo angles vs time (3 motors)")
    plt.plot(t, s1, label="servo 1 (deg)")
    plt.plot(t, s2, label="servo 2 (deg)")
    plt.plot(t, s3, label="servo 3 (deg)")
    plt.xlabel("time (s)")
    plt.ylabel("angle (deg)")
    plt.legend()
    plt.grid(True)

    plt.show()

def main():
    global sim_running
    running = True
    while running:
        dt = clock.tick(60) / 1000.0  

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mx, my = event.pos
                    kp_slider.handle_event(event)
                    ki_slider.handle_event(event)
                    kd_slider.handle_event(event)
                    if not (kp_slider.rect.collidepoint((mx, my)) or
                            ki_slider.rect.collidepoint((mx, my)) or
                            kd_slider.rect.collidepoint((mx, my))):
                        cx, cy = platform_center_px
                        if (mx - cx) ** 2 + (my - cy) ** 2 <= platform_radius_px ** 2:
                            reset_sim_to_click(mx, my)
            elif event.type in (pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION):
                kp_slider.handle_event(event)
                ki_slider.handle_event(event)
                kd_slider.handle_event(event)
        Kp = kp_slider.value
        Ki = ki_slider.value
        Kd = kd_slider.value

        if sim_running:
            step_sim(dt, Kp, Ki, Kd)

        draw()

    pygame.quit()
    show_plots()

if __name__ == "__main__":
    main()
