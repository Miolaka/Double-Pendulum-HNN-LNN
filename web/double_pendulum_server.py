import torch
import numpy as np
import asyncio
import json
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI()


# ------------------------------------------------------------
# Physics (GPU-accelerated)
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


def rk4_step(state, L1, L2, m1, m2, g, dt):
    k1 = deriv_torch(state, L1, L2, m1, m2, g)
    k2 = deriv_torch(state + 0.5 * dt * k1, L1, L2, m1, m2, g)
    k3 = deriv_torch(state + 0.5 * dt * k2, L1, L2, m1, m2, g)
    k4 = deriv_torch(state + dt * k3, L1, L2, m1, m2, g)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


class DoublePendulum:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.L1 = self.L2 = 1.0
        self.m1 = self.m2 = 1.0
        self.g = 9.81
        self.dt = 0.01
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
        self.state = torch.tensor([np.radians(120), 0.0, np.radians(-10), 0.0],
                                  dtype=torch.float32, device=self.device)
        self.t = 0.0


# ------------------------------------------------------------
# WebSocket live connection
# ------------------------------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    sim = DoublePendulum()
    print("WebSocket connected.")
    try:
        while True:
            sim.step()
            x1, y1, x2, y2 = sim.positions()
            await ws.send_text(json.dumps({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "t": sim.t
            }))
            await asyncio.sleep(0.01)  # ~100 FPS
    except Exception as e:
        print("WebSocket closed:", e)
    finally:
        await ws.close()


# ------------------------------------------------------------
# Simple static page
# ------------------------------------------------------------
@app.get("/")
async def index():
    return HTMLResponse(open("double_pendulum_client.html").read())


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
