import sys, json, time
import torch, numpy as np
import webview


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


def rk4_step(state, L1, L2, m1, m2, g, dt):
    k1 = deriv_torch(state, L1, L2, m1, m2, g)
    k2 = deriv_torch(state + 0.5 * dt * k1, L1, L2, m1, m2, g)
    k3 = deriv_torch(state + 0.5 * dt * k2, L1, L2, m1, m2, g)
    k4 = deriv_torch(state + dt * k3, L1, L2, m1, m2, g)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


class DoublePendulum:
    def __init__(self, device='cuda'):
        self.L1 = self.L2 = 1.0
        self.m1 = self.m2 = 1.0
        self.g = 9.81
        self.dt = 0.01
        self.device = torch.device(device)
        self.reset()
        self.running = True

    def step(self):
        self.state = rk4_step(self.state, self.L1, self.L2,
                              self.m1, self.m2, self.g, self.dt)
        self.t += self.dt

    def get_positions(self):
        θ1, ω1, θ2, ω2 = self.state
        x1 = self.L1 * torch.sin(θ1)
        y1 = -self.L1 * torch.cos(θ1)
        x2 = x1 + self.L2 * torch.sin(θ2)
        y2 = y1 - self.L2 * torch.cos(θ2)
        return float(x1), float(y1), float(x2), float(y2)

    def reset(self):
        self.state = torch.tensor([np.radians(120), 0.0,
                                   np.radians(-10), 0.0],
                                   dtype=torch.float32, device=self.device)
        self.t = 0.0


class API:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sim = DoublePendulum(device=device)

    def toggle(self):
        self.sim.running = not self.sim.running
        return json.dumps({"running": self.sim.running})

    def reset(self):
        self.sim.reset()
        return "reset"

    def get_state(self):
        sim = self.sim
        if sim.running:
            sim.step()
        x1, y1, x2, y2 = sim.get_positions()
        t = float(sim.t)
        return json.dumps({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "t": t})


html_page = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>Double Pendulum GPU</title>
<style>
body { margin:0; overflow:hidden; background:#111; color:white; text-align:center; font-family:sans-serif; }
canvas { display:block; margin:auto; background:black; }
button { margin:8px; padding:8px 14px; font-size:16px; border-radius:6px; cursor:pointer; }
</style>
</head>
<body>
<h3>Double Pendulum (GPU + PyWebView)</h3>
<canvas id="canvas" width="800" height="800"></canvas><br>
<button onclick="toggle()">▶ / ❚❚</button>
<button onclick="reset()">Reset</button>

<script>
let ctx = document.getElementById('canvas').getContext('2d');

async function frame(){
    try {
        const s = JSON.parse(await pywebview.api.get_state());
        ctx.fillStyle = 'rgba(0,0,0,0.25)';
        ctx.fillRect(0,0,800,800);

        const ox = 400, oy = 150, scale = 250;
        ctx.strokeStyle = 'cyan';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(ox, oy);
        ctx.lineTo(ox + s.x1*scale, oy + s.y1*scale);
        ctx.lineTo(ox + s.x2*scale, oy + s.y2*scale);
        ctx.stroke();

        ctx.fillStyle = 'red';
        ctx.beginPath();
        ctx.arc(ox + s.x2*scale, oy + s.y2*scale, 6, 0, 2*Math.PI);
        ctx.fill();
    } catch (e) {
        console.error("Error fetching state:", e);
    }

    setTimeout(frame, 1000/60);
}

// Start immediately
frame();

function toggle(){ pywebview.api.toggle(); }
function reset(){ pywebview.api.reset(); }
</script>
</body>
</html>
"""

if __name__ == "__main__":
    api = API()
    webview.create_window("GPU Double Pendulum", html=html_page, js_api=api,
                          width=820, height=900)
    webview.start()
