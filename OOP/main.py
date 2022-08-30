import numpy as np
import dryden_model
import matplotlib.pyplot as plt
from operator import add
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import colors as mcolors

#######
##
# NOTE: Set N , 10 for a 3D plot and N > 10 for a 2D plot
##
######


class Windfield:
    def __init__(self):
        self.t_p = None
        self.t_sim = None
        self.nr_samples = None
        self.h = None
        self.a = None
        self.ws_magn = None
        self.ws_dir = None
        self.y1_const = None
        self.y2_const = None

        self.y1_dyn = None
        self.y2_dyn = None
        self.y3_dyn = None
        self.y1_combined = None
        self.y2_combined = None
        self.y3_combined = None

        # Set simulation parameters
        self.set_simulation_parameters()

        # Compute stuff
        self.compute_turbulence()
        self.compute_constant_wind()
        self.compute_total_wind()

    def set_simulation_parameters(self):
        self.t_sim = 20
        self.nr_samples = 20
        self.h = float(input("please enter altitude in m: ") or 10)  # "or" ensures defaults
        self.a = float(input("please enter drone's airspeed in m/s: ") or 1)
        self.ws_magn = float(input("please enter the constant true windspeed magnitude in m/s: ") or 3)
        self.ws_dir = float(input("please enter the constant true windspeed direction in deg: ") or 0)
        self.ws_dir = self.ws_dir * np.pi / 180.

    def compute_turbulence(self):
        # Compute turbulence  gusts
        self.t_p, self.y1_dyn, self.y2_dyn, self.y3_dyn, self.nr_samples = \
            dryden_model.dryden_wind_velocities(self.h, self.a, self.t_sim, self.nr_samples)

    def compute_constant_wind(self):
        # Compute the x direction
        y1_const = - self.ws_magn * np.cos(self.ws_dir)
        self.y1_const = [y1_const for _ in range(len(self.t_p))]        # create a whole list of this constant number

        # Compute y direction
        y2_const = - self.ws_magn * np.sin(self.ws_dir)
        self.y2_const = [y2_const for _ in range(len(self.t_p))]

        # do not alter z

    def compute_total_wind(self):
        # Add the components of constant and turbulence
        self.y1_combined = list(map(add, self.y1_dyn, self.y1_const))     # Add the constant wind speed component to the gusts
        self.y2_combined = list(map(add, self.y2_dyn, self.y2_const))     # Add the constant wind speed component to the gusts
        self.y3_combined = self.y3_dyn


class Main:
    def __init__(self):
        print("\n **** Simplified airflow simulator **** \n")

        # Simulate the windfields
        windfield = Windfield()

        # Plot in 2D or 3D, depending on how many data points
        if windfield.nr_samples > 10:
            self.plot_graphs_2d(windfield)
        else:
            self.plot_graphs_3d(windfield)

        # Save your data in a csv
        self.save_data(windfield)

    # noinspection PyMethodMayBeStatic
    def plot_graphs_2d(self, windfield):
        # Graph out wind velocities in subplots
        fig, axs = plt.subplots(3, 1)
        fig.suptitle('Simulated wind in x, y and z direction', fontsize=12)

        axs[0].plot(windfield.t_p, windfield.y1_combined, 'b')
        axs[0].set_xlabel('time')
        axs[0].set_ylabel('along-wind (x-dir) in m/s (P)')
        axs[0].grid(True)
        # axs[0].title.set_text('First Plot')
        # axs[0].set_ylim(-10, 10)                   # Zoom out a bit

        axs[1].plot(windfield.t_p, windfield.y2_combined, 'r')
        axs[1].set_xlabel('time')
        axs[1].set_ylabel('cross-wind (y-dir) in m/s (P)')
        axs[1].grid(True)

        axs[2].plot(windfield.t_p, windfield.y3_combined, 'g')
        axs[2].set_xlabel('time in s')
        axs[2].set_ylabel('vertical-wind (z-dir) in m/s (P)')
        axs[2].grid(True)

        # Show all plots
        fig.tight_layout()
        plt.show()

    # noinspection PyMethodMayBeStatic
    def plot_graphs_3d(self, windfield):
        # NOTE: Only works if you have N < 10
        # plot 3D figure
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlabel('wind in x-direction')
        ax.set_ylabel('wind in y-direction')
        ax.set_zlabel('wind in z-direction')
        ax.set_title('3D plot of simulated windfields')
        # see documentation for more: https://matplotlib.org/devdocs/api/_as_gen/mpl_toolkits.mplot3d.axes3d.Axes3D.html

        # Make the grid
        x, y, z = np.meshgrid(np.linspace(-2.0, 2.0, windfield.nr_samples),
                              np.linspace(-2.0, 2.0, windfield.nr_samples),
                              np.linspace(-2.0, 2.0, windfield.nr_samples))

        # Make the direction data for the arrows
        u = windfield.y1_combined
        v = windfield.y2_combined
        w = windfield.y3_combined

        colors = [mcolors.to_rgba(c) for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]
        ax.quiver(x, y, z, u, v, w, length=0.3, normalize=True, color=colors)
        plt.show()

    # noinspection PyMethodMayBeStatic
    def save_data(self, windfield):
        zipped = list(zip(windfield.t_p,
                          windfield.y1_combined,
                          windfield.y2_combined,
                          windfield.y3_combined))
        df = pd.DataFrame(zipped, columns=['time', 'wind_x_dir', 'wind_y_dir', 'wind_z_dir'])
        print(df.head())
        df.to_csv(f'data_generated/windfield_data_T{str(windfield.t_sim)}_h{str(int(windfield.h))}_wsmag{str((int(windfield.ws_magn)))}.csv')


if __name__ == "__main__":
    main = Main()


