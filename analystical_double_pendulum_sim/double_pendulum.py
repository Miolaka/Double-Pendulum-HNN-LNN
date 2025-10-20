import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, CheckButtons
from scipy.integrate import solve_ivp

# --- Double Pendulum Physics ---
def deriv(t, state, L1, L2, m1, m2, g):
    θ1, ω1, θ2, ω2 = state
    Δ = θ2 - θ1

    den1 = (m1 + m2) * L1 - m2 * L1 * np.cos(Δ)**2
    den2 = (L2 / L1) * den1

    ω1_dot = (m2 * L1 * ω1**2 * np.sin(Δ) * np.cos(Δ)
              + m2 * g * np.sin(θ2) * np.cos(Δ)
              + m2 * L2 * ω2**2 * np.sin(Δ)
              - (m1 + m2) * g * np.sin(θ1)) / den1

    ω2_dot = (-m2 * L2 * ω2**2 * np.sin(Δ) * np.cos(Δ)
              + (m1 + m2) * (g * np.sin(θ1) * np.cos(Δ)
              - L1 * ω1**2 * np.sin(Δ) - g * np.sin(θ2))) / den2

    return [ω1, ω1_dot, ω2, ω2_dot]


# --- Simulation Wrapper ---
class DoublePendulumSim:
    def __init__(self):
        self.L1 = 1.0
        self.L2 = 1.0
        self.m1 = 1.0
        self.m2 = 1.0
        self.g = 9.81
        self.state = np.radians([120, 0, -10, 0])
        self.dt = 0.02
        self.running = False
        self.solver = "RK4"

    def step(self):
        if self.solver == "RK4":
            self.state = self._rk4_step()
        else:
            self.state = self._euler_step()

    def _rk4_step(self):
        s = self.state
        f = lambda t, y: deriv(t, y, self.L1, self.L2, self.m1, self.m2, self.g)
        k1 = np.array(f(0, s))
        k2 = np.array(f(0, s + 0.5*self.dt*k1))
        k3 = np.array(f(0, s + 0.5*self.dt*k2))
        k4 = np.array(f(0, s + self.dt*k3))
        return s + (self.dt/6)*(k1 + 2*k2 + 2*k3 + k4)

    def _euler_step(self):
        f = deriv(0, self.state, self.L1, self.L2, self.m1, self.m2, self.g)
        return self.state + self.dt * np.array(f)

    def positions(self):
        θ1, ω1, θ2, ω2 = self.state
        x1 = self.L1 * np.sin(θ1)
        y1 = -self.L1 * np.cos(θ1)
        x2 = x1 + self.L2 * np.sin(θ2)
        y2 = y1 - self.L2 * np.cos(θ2)
        return x1, y1, x2, y2


# --- GUI / Animation ---
def run_gui():
    sim = DoublePendulumSim()

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.35)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")
    ax.set_title("Double Pendulum")

    (line,) = ax.plot([], [], "o-", lw=3, color="blue")
    time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes)

    # --- Sliders ---
    axcolor = "lightgoldenrodyellow"
    axL1 = plt.axes([0.25, 0.25, 0.65, 0.03], facecolor=axcolor)
    axL2 = plt.axes([0.25, 0.20, 0.65, 0.03], facecolor=axcolor)
    axM1 = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
    axM2 = plt.axes([0.25, 0.10, 0.65, 0.03], facecolor=axcolor)
    axG  = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)

    sL1 = Slider(axL1, "rod-1 length", 0.5, 2.0, valinit=1.0)
    sL2 = Slider(axL2, "rod-2 length", 0.5, 2.0, valinit=1.0)
    sM1 = Slider(axM1, "mass-1", 0.5, 3.0, valinit=1.0)
    sM2 = Slider(axM2, "mass-2", 0.5, 3.0, valinit=1.0)
    sG  = Slider(axG, "gravity", 0.1, 15.0, valinit=9.81)

    # --- Buttons ---
    start_ax = plt.axes([0.05, 0.05, 0.1, 0.04])
    reset_ax = plt.axes([0.05, 0.11, 0.1, 0.04])
    b_start = Button(start_ax, "▶ / ❚❚")
    b_reset = Button(reset_ax, "Reset")

    def update_sliders(val=None):
        sim.L1, sim.L2, sim.m1, sim.m2, sim.g = sL1.val, sL2.val, sM1.val, sM2.val, sG.val

    def toggle_run(event):
        sim.running = not sim.running

    def reset(event):
        sim.state = np.radians([120, 0, -10, 0])
        sim.running = False
        update(0)

    sL1.on_changed(update_sliders)
    sL2.on_changed(update_sliders)
    sM1.on_changed(update_sliders)
    sM2.on_changed(update_sliders)
    sG.on_changed(update_sliders)
    b_start.on_clicked(toggle_run)
    b_reset.on_clicked(reset)

    # --- Animation Update ---
    def update(frame):
        if sim.running:
            sim.step()
        x1, y1, x2, y2 = sim.positions()
        line.set_data([0, x1, x2], [0, y1, y2])
        time_text.set_text(f"t = {frame*sim.dt:.2f}s")
        return line, time_text

    ani = FuncAnimation(fig, update, interval=20, blit=False)
    plt.show()

if __name__ == "__main__":
    run_gui()
