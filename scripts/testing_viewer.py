
import mujoco
import mujoco_viewer
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/prakyath/github/personal/test_walking_with_palm')
from utils.integration import AccIntegration




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
    store_dat = []

    AccInt = AccIntegration()




    pos = np.zeros((3, 1))

    acc = []
    for i in range(10000):
        #data.ctrl[7] = -0.8
        if i > 200:
            data.ctrl[7] = 0.2
            data.ctrl[8] = 0.2
        if i > 600:
            data.ctrl[7] = -0.8
        acc.append(data.sensordata[:3].reshape(3,1))
        if len(acc) > 100:
            acc_arr = np.array(acc)[:, :, 0].T
            acc_arr = acc_arr * 10e10
            acc_arr = acc_arr.round(2)
            acc_arr[2,:] -= 9.81 * 10e10
            print(data.body("torso").xpos)
            position = AccInt.predict(acc_arr).reshape(3,1)
            pos = np.hstack((pos, position))

        if viewer.is_alive:
            mujoco.mj_step(model, data) #type: ignore
            viewer.render()

        else:
            break

    plt.plot(pos[0, :], label = 'x')
    plt.plot(pos[1, :], label = 'y')
    plt.plot(pos[2, :], label = 'z')
    plt.legend()
    plt.show()
    # close
    viewer.close()
