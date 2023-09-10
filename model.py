from phi import flow
import matplotlib.pyplot as plt
from tqdm import tqdm


def main():
    V = flow.StaggeredGrid(
        values=(0.0, 0.0),  # initially stagnant fluid
        extrapolation=0.0,  # BC
        x=64,
        y=64,
        bounds=flow.Box(x=100, y=100),
    )
    smoke = flow.CenteredGrid(
        values=0.0,
        extrapolation=flow.extrapolation.BOUNDARY,
        x=200,
        y=200,
        bounds=flow.Box(x=100, y=100),
    )
    inflow = flow.CenteredGrid(
        values=flow.SoftGeometryMask(flow.Sphere(x=50, y=9.5, radius=5)),
        extrapolation=0.0,
        bounds=smoke.bounds,
        resolution=smoke.resolution,
    )

    def step(v_prev, smoke_prev, dt=1.0):
        smoke_next = flow.advect.mac_cormack(smoke_prev, v_prev, dt) + inflow
        bouyancy = smoke_next * (0.0, 0.1) @ V
        V_step = flow.advect.semi_lagrangian(v_prev, v_prev, dt) + bouyancy * dt
        V_next, pressure = flow.fluid.make_incompressible(V_step)

        return V_next, smoke_next

    plt.style.use("dark_background")
    for _ in tqdm(range(150)):
        V, smoke = step(V, smoke)
        smoke_val = smoke.values.numpy("y,x")
        plt.imshow(smoke_val, origin="lower")
        plt.draw()
        plt.pause(0.01)
        plt.clf()


if __name__ == "__main__":
    main()
