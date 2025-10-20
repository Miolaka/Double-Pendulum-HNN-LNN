import sys
import torch
import numpy as np
from vispy import scene, app
from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt



# ------------------------------------------------------------
# Physics (PyTorch RK4 on GPU)
# ------------------------------------------------------------
def deriv_torch(state, L1, L2, m1, m2, g):
    θ1, ω1, θ2, ω2 = state
    delta = θ2 - θ1
    den1 = (m1 + m2) * L1 - m2 * L1 * torch.cos(delta)**2
    den2 = (L2 / L1) * den1

    ω1_dot = (m2 * L1 * ω1**2 * torch.sin(delta) * torch.cos(delta)
              + m2 * g * torch.sin(θ2) * torch.cos(delta)
              + m2 * L2 * ω2**2 * torch.sin(delta)
              - (m1 + m2) * g * torch.sin(θ1)) / den1

    ω2_dot = (-m2 * L2 * ω2**2 * torch.sin(delta) * torch.cos(delta)
              + (m1 + m2) * (g * torch.sin(θ1) * torch.cos(delta)
              - L1 * ω1**2 * torch.sin(delta) - g * torch.sin(θ2))) / den2

    return torch.stack([ω1, ω1_dot, ω2, ω2_dot])


def rk4_step(state, L1, L2, m1, m2, g, dt):
    k1 = deriv_torch(state, L1, L2, m1, m2, g)
    k2 = deriv_torch(state + 0.5 * dt * k1, L1, L2, m1, m2, g)
    k3 = deriv_torch(state + 0.5 * dt * k2, L1, L2, m1, m2, g)
    k4 = deriv_torch(state + dt * k3, L1, L2, m1, m2, g)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


# ------------------------------------------------------------
# Simulation logic
# ------------------------------------------------------------
class DoublePendulumTorch:
    def __init__(self, device='cuda'):
        self.L1, self.L2 = 1.0, 1.0
        self.m1, self.m2 = 1.0, 1.0
        self.g = 9.81
        self.dt = 0.01
        self.device = torch.device(device)
        self.reset()

    def step(self):
        self.state = rk4_step(self.state, self.L1, self.L2, self.m1, self.m2, self.g, self.dt)
        self.t += self.dt

    def positions(self):
        θ1, ω1, θ2, ω2 = self.state
        x1 = self.L1 * torch.sin(θ1)
        y1 = -self.L1 * torch.cos(θ1)
        x2 = x1 + self.L2 * torch.sin(θ2)
        y2 = y1 - self.L2 * torch.cos(θ2)
        return float(x1), float(y1), float(x2), float(y2)

    def reset(self):
        self.state = torch.tensor(
            [np.radians(30), 0.0, np.radians(15), 0.0],
            dtype=torch.float32, device=self.device
        )
        self.t = 0.0


# ------------------------------------------------------------
# VisPy + Qt Integration
# ------------------------------------------------------------
class DoublePendulumApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Double Pendulum (GPU + VisPy + PyTorch)")

        # Simulation
        self.sim = DoublePendulumTorch(device='cuda' if torch.cuda.is_available() else 'cpu')
        self.running = False

        # Create VisPy canvas
        self.canvas = scene.SceneCanvas(keys='interactive', bgcolor='white', size=(600, 600), show=True)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.PanZoomCamera(rect=(-2.2, -2.2, 4.4, 4.4))
        self.view.camera.aspect = 1.0

        self.line = scene.Line(pos=np.zeros((3, 2)), color='blue', width=4, parent=self.view.scene)
        self.mass1 = scene.Markers(pos=np.zeros((1, 2)), face_color='red', size=12, parent=self.view.scene)
        self.mass2 = scene.Markers(pos=np.zeros((1, 2)), face_color='green', size=12, parent=self.view.scene)

        # Layout
        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.addWidget(self.canvas.native)
        self.setCentralWidget(central)

        # Add sliders
        self.add_sliders(layout)

        # Timer
        self.timer = app.Timer(interval='auto', connect=self.on_timer, start=True)

    def add_sliders(self, layout):
        grid = QtWidgets.QGridLayout()

        self.sliders = {}
        for i, (name, init, lo, hi) in enumerate([
            ("L1", 1.0, 0.1, 5.0),
            ("L2", 1.0, 0.1, 5.0),
            ("m1", 1.0, 0.1, 10.0),
            ("m2", 1.0, 0.1, 10.0),
            ("g", 9.81, 1.0, 15.0),
        ]):
            label = QtWidgets.QLabel(f"{name}: {init:.2f}")
            slider = QtWidgets.QSlider()
            slider.setOrientation(Qt.Orientation.Horizontal)

            slider.setMinimum(0)
            slider.setMaximum(100)
            slider.setValue(int((init - lo) / (hi - lo) * 100))
            slider.valueChanged.connect(lambda val, n=name, l=label, low=lo, high=hi: self.on_slider_change(val, n, l, low, high))
            grid.addWidget(label, i, 0)
            grid.addWidget(slider, i, 1)
            self.sliders[name] = (slider, lo, hi)

        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()
        start_btn = QtWidgets.QPushButton("▶ / ❚❚")
        start_btn.clicked.connect(self.toggle_run)
        reset_btn = QtWidgets.QPushButton("Reset")
        reset_btn.clicked.connect(self.reset)
        btn_layout.addWidget(start_btn)
        btn_layout.addWidget(reset_btn)

        layout.addLayout(grid)
        layout.addLayout(btn_layout)

    def on_slider_change(self, val, name, label, lo, hi):
        valf = lo + (hi - lo) * val / 100
        setattr(self.sim, name, valf)
        label.setText(f"{name}: {valf:.2f}")

    def toggle_run(self):
        self.running = not self.running

    def reset(self):
        self.sim.reset()
        self.running = False

    def on_timer(self, event):
        if self.running:
            self.sim.step()
        x1, y1, x2, y2 = self.sim.positions()
        pts = np.array([[0, 0], [x1, y1], [x2, y2]], dtype=np.float32)
        self.line.set_data(pos=pts)
        self.mass1.set_data(pos=pts[1:2])
        self.mass2.set_data(pos=pts[2:3])
        self.canvas.update()


# ------------------------------------------------------------
# Run the app
# ------------------------------------------------------------
if __name__ == "__main__":
    app.use_app('pyqt6')
    qt_app = QtWidgets.QApplication(sys.argv)
    win = DoublePendulumApp()
    win.show()
    qt_app.exec()
