import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button


# ------------------------------------------------------------
# Exact double-pendulum equations (PyTorch)
# ------------------------------------------------------------
def deriv_torch(state, L1, L2, m1, m2, g):
    θ1, ω1, θ2, ω2 = state
    Δ = θ2 - θ1

    den1 = (m1 + m2) * L1 - m2 * L1 * torch.cos(Δ)**2
    den2 = (L2 / L1) * den1

    ω1_dot = (m2 * L1 * ω1**2 * torch.sin(Δ) * torch.cos(Δ)
              + m2 * g * torch.sin(θ2) * torch.cos(Δ)
              + m2 * L2 * ω2**2 * torch.sin(Δ)
              - (m1 + m2) * g * torch.sin(θ1)) / den1

    ω2_dot = (-m2 * L2 * ω2**2 * torch.sin(Δ) * torch.cos(Δ)
              + (m1 + m2) * (g * torch.sin(θ1) * torch.cos(Δ)
              - L1 * ω1**2 * torch.sin(Δ) - g * torch.sin(θ2))) / den2

    return torch.stack([ω1, ω1_dot, ω2, ω2_dot])


# ------------------------------------------------------------
# Torch-based simulator
# ------------------------------------------------------------
class DoublePendulumTorch:
    def __init__(self, device='cpu'):
        # default parameters
        self.L1, self.L2 = 1.0, 1.0
        self.m1, self.m2 = 1.0, 1.0
        self.g = 9.81
        self.dt = 0.05
        self.device = torch.device(device)

        # initial state [θ1, ω1, θ2, ω2]
        self.state = torch.tensor([np.radians(10), 0.0, 0.0, 0.0],
                                  dtype=torch.float32, device=self.device)
        self.t = 0.0
        self.running = False

    # Runge–Kutta 4 integrator (stable + fast)
    def rk4_step(self):
        s = self.state
        L1, L2, m1, m2, g, dt = self.L1, self.L2, self.m1, self.m2, self.g, self.dt

        k1 = deriv_torch(s, L1, L2, m1, m2, g)
        k2 = deriv_torch(s + 0.5 * dt * k1, L1, L2, m1, m2, g)
        k3 = deriv_torch(s + 0.5 * dt * k2, L1, L2, m1, m2, g)
        k4 = deriv_torch(s + dt * k3, L1, L2, m1, m2, g)

        self.state = s + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        self.t += dt

    def positions(self):
        θ1, ω1, θ2, ω2 = self.state
        x1 = self.L1 * torch.sin(θ1)
        y1 = -self.L1 * torch.cos(θ1)
        x2 = x1 + self.L2 * torch.sin(θ2)
        y2 = y1 - self.L2 * torch.cos(θ2)
        return x1.item(), y1.item(), x2.item(), y2.item()


# ------------------------------------------------------------
# Interactive animation GUI
# ------------------------------------------------------------
def run_live():
    sim = DoublePendulumTorch(device='cuda' if torch.cuda.is_available() else 'cpu')

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.35)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.set_title("Live Double Pendulum (PyTorch RK4)")

    line, = ax.plot([], [], 'o-', lw=3, color='blue')
    time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes)

    # sliders
    axcolor = 'lightgoldenrodyellow'
    axL1 = plt.axes([0.25, 0.25, 0.65, 0.03], facecolor=axcolor)
    axL2 = plt.axes([0.25, 0.20, 0.65, 0.03], facecolor=axcolor)
    axM1 = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
    axM2 = plt.axes([0.25, 0.10, 0.65, 0.03], facecolor=axcolor)
    axG  = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)

    sL1 = Slider(axL1, "rod-1 length", 0.1, 5.0, valinit=1.0)
    sL2 = Slider(axL2, "rod-2 length", 0.1, 5.0, valinit=1.0)
    sM1 = Slider(axM1, "mass-1", 0.0, 10.0, valinit=1.0)
    sM2 = Slider(axM2, "mass-2", 0.0, 10.0, valinit=1.0)
    sG  = Slider(axG, "gravity", 1.0, 15.0, valinit=9.81)

    def update_params(val=None):
        sim.L1, sim.L2, sim.m1, sim.m2, sim.g = \
            sL1.val, sL2.val, sM1.val, sM2.val, sG.val

    sL1.on_changed(update_params)
    sL2.on_changed(update_params)
    sM1.on_changed(update_params)
    sM2.on_changed(update_params)
    sG.on_changed(update_params)

    # buttons
    start_ax = plt.axes([0.05, 0.05, 0.1, 0.04])
    reset_ax = plt.axes([0.05, 0.11, 0.1, 0.04])
    b_start = Button(start_ax, "▶ / ❚❚")
    b_reset = Button(reset_ax, "Reset")

    def toggle_run(event):
        sim.running = not sim.running

    def reset(event):
        sim.state = torch.tensor([np.radians(10), 0.0, 0.0, 0.0],
                                 dtype=torch.float32, device=sim.device)
        sim.t = 0.0
        sim.running = False
        update(0)

    b_start.on_clicked(toggle_run)
    b_reset.on_clicked(reset)

    # animation update
    def update(frame):
        if sim.running:
            sim.rk4_step()
        x1, y1, x2, y2 = sim.positions()
        line.set_data([0, x1, x2], [0, y1, y2])
        time_text.set_text(f"t = {sim.t:.2f} s")
        return line, time_text

    anim = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    run_live()
