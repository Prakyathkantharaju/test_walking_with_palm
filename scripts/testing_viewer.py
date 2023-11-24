
import mujoco #type: reportMissingImports=false
import mujoco_viewer
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import scipy.signal as signal




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")

    model_path = parser.parse_args().model_path

    model = mujoco.MjModel.from_xml_path(model_path) #type: ignore
    data = mujoco.MjData(model) #type: ignore

    # create the viewer object
    viewer = mujoco_viewer.MujocoViewer(model, data)

    # simulate and render
    angle = 0.00
    store_data = []


    def _integration_acc_to_position(acc, prev_pos, prev_vel):
        dt = 0.01
        vel = integrate.trapezoid(acc, dx=dt, axis=0)
        prev_vel = np.concatenate([prev_vel, vel.reshape(1, 3)], axis=0)
        pos = integrate.trapezoid(prev_vel, dx=dt, axis=0)
        prev_pos = np.concatenate([prev_pos.reshape(-1,3), pos.reshape(1, 3)], axis = 0)
        return prev_vel, prev_pos

    pos = np.zeros((1, 3))
    vel = np.zeros((1, 3))
    acc = [] 
    for i in range(10000):
        #data.ctrl[7] = -0.8
        # print(data.sensordata[3:]) # expectation is 6
        acc.append(data.sensordata[:3])
        if len(acc) > 10:
            acc_arr = np.array(acc)
            vel, pos = _integration_acc_to_position(acc_arr, pos, vel)
            pos = signal.detrend(pos)
            vel = signal.detrend(vel)
            print(signal.detrend(pos)[-1,:])
            acc = []
        
        if viewer.is_alive:
            mujoco.mj_step(model, data) #type: ignore
            viewer.render()
        else:
            break

    print(pos.shape)
    plt.plot()
    plt.show(signal.detrend(pos))
    # close
    viewer.close()
