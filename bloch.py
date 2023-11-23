"""
All times are time values inside the simulated world.
All units are S.I.

TODO:
- [ ] Implement: RF excitation and spin relaxation
"""

from pathlib import Path
import numpy as np
import pandas as pd
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
import matplotlib.pyplot as plt
from tqdm import tqdm



# Constants
PI = np.pi
GAMMA = 42.57638474 * 1e6  # Gyromagnetic ratio of 1H in H20 [Hz/T]
GAMMA_BY_2_PI = GAMMA / (2*PI)

# Spin system config
M0 = resize(shepp_logan_phantom(), (16, 16), anti_aliasing=True)

T1 = None
T2 = None
VOXEL_SIZE = np.array((1e-3, 1e-3, 1e-3))  # [m]

# Scanner config
TR = 5e-3  # [s]
TE = None  # [s]
FA = PI / 2  # [rad]
START_TIME_GY = 1e-3   # Local time relative to the rep [s]
DURATION_GY = 1e-3  # [s]
START_TIME_GX = 3e-3   # Local time relative to the rep [s]
DURATION_GX = 1e-3  # [s]

# Simulation config
OUTPUT_DIR = Path("C:/Users/csrao/git-personal/comp-img-garage/algorithms/mri/bloch/output/")
TIME_STEP = 2e-6  # [s]
SIMULATE_RELAXATION = False
SIMULATE_B1_WAVEFORM = False  # `False`: use a rotation op with given FA; `True`: construct and apply a pulse waveform



class SpinSystem:

    def __init__(self, param_maps, voxel_size):        
        
        self.param_maps = param_maps  # Dict of meshgrid arrays
        self.voxel_size = voxel_size  # (x,y), [m]
        self.fov = np.array([param_maps['m0'].shape[1] * voxel_size[1] for i in range(2)])
        
        # Calc voxel center positions
        coord_values_x = np.arange(-self.fov[0] / 2, self.fov[0] / 2, voxel_size[0])
        coord_values_y = np.arange(-self.fov[1] / 2, self.fov[1] / 2, voxel_size[1])
        self.coords_x, self.coords_y = np.meshgrid(coord_values_x, coord_values_y)

        # Magnetization array
        self.m_state = param_maps['m0'] * np.exp(1j * 0)
    
    def apply_gradient(self, gradient):
        gx, gy = gradient[0], gradient[1]
        bz_field = gx * self.coords_x + gy * self.coords_y
        phase = GAMMA * bz_field * TIME_STEP
        self.m_state = self.m_state * np.exp(1j * phase)

    def observe(self):
        return self.m_state.sum()


class Scanner:

    def __init__(self, spin_system, seq_config, clock, recorder):
        
        self.spin_system = spin_system
        self.seq_config = seq_config
        self.clock = clock
        self.recorder = recorder
        
        self.kspace_data = np.zeros_like(spin_system.m_state)
        
        # Calc k-value coordinates
        self.kvoxel_size = 1. / spin_system.fov
        self.kfov = 1. / spin_system.voxel_size       
        kcoord_values_x = np.arange(-self.kfov[0] / 2., self.kfov[0] / 2., self.kvoxel_size[0])
        kcoord_values_y = np.arange(-self.kfov[1] / 2., self.kfov[1] / 2., self.kvoxel_size[1])
        self.kcoords_x, self.kcoords_y = np.meshgrid(kcoord_values_x, kcoord_values_y)

        # Tolerance used to detect when a readout k-point in the kcoords grid is reached. 
        # Needed because of the necessary time discretization used in the simulation (via the non-zero time step).
        # Definition: half the distance in kspace traversed per time step
        self.kcoord_tol = 0.5 * (self.kfov[0] / self.seq_config['duration_gx']) * self.clock.time_step

        self.gradient_history = []
        self.full_traj = []
        self.readout_traj = []

    def apply_rf(self, flip_angle):
        pass

    def apply_gradient(self, gradient, readout=False):
        
        # Manipulate the spin system with the gradient
        self.spin_system.apply_gradient(gradient)
        self.gradient_history.append(gradient)

        # Estimate the k-coords corresponding to the spin system's phase
        kcoord_estim = GAMMA_BY_2_PI * np.array(self.gradient_history).sum(axis=0) * self.clock.time_step        
        self.full_traj.append(kcoord_estim)

        # Readout
        if readout:
            l2_distances = np.linalg.norm(np.stack([self.kcoords_x - kcoord_estim[0], self.kcoords_y - kcoord_estim[1]], axis=2), ord=2, axis=2)
            if np.any(l2_distances < self.kcoord_tol):
                closest_kcoord_x = self.kcoords_x.flatten()[np.argmin(l2_distances)]
                closest_kcoord_y = self.kcoords_y.flatten()[np.argmin(l2_distances)]
                ksample_idx = np.argwhere((self.kcoords_x == closest_kcoord_x) & (self.kcoords_y == closest_kcoord_y)).squeeze()
                ksample = self.spin_system.observe()  # Observe the encoded data
                print(ksample, kcoord_estim, ksample_idx)  # DEBUG LOG
                self.kspace_data[ksample_idx[0], ksample_idx[1]] = ksample
                self.readout_traj.append(kcoord_estim)

                self.recorder.set_m_state(self.spin_system.m_state)
                self.recorder.set_kspace_data(self.kspace_data)
                self.recorder.save_world_state()

    def reset_kspace_coord(self):
        kcoord_curr = [0., 0.]
        kcoord_target = self.kcoords_x[0, 0], self.kcoords_y[0, 0]
        time_to_reset = self.clock.time_step 
        grad_value_x = (kcoord_target[0] - kcoord_curr[0]) / (GAMMA_BY_2_PI * time_to_reset)
        grad_value_y = (kcoord_target[1] - kcoord_curr[1]) / (GAMMA_BY_2_PI * time_to_reset)
        gradient = np.array([grad_value_x, grad_value_y])
        self.apply_gradient(gradient)

    def construct_seq_repr(self, rep_idx):

        num_time_steps_per_rep = round(self.seq_config['tr'] / self.clock.time_step)
        rep_local_time_values = np.arange(0, self.seq_config['tr'], self.clock.time_step)

        # Init seq repr
        rf_values = np.zeros((num_time_steps_per_rep,))        
        gx_values = gy_values = np.zeros((num_time_steps_per_rep,))
        seq_repr = {'t': rep_local_time_values, 'fa': rf_values, 'gx': gx_values, 'gy': gy_values}
        seq_repr = pd.DataFrame.from_dict(seq_repr)
        
        # Populate seq repr
        seq_repr['fa'][0] = self.seq_config['fa']

        num_freq_encodes, num_phase_encodes = self.kspace_data.shape[0], self.kspace_data.shape[1]
        gx_peak = self.kfov[0] / (GAMMA_BY_2_PI * self.seq_config['duration_gx'])
        gy_peak = self.kfov[1] / (GAMMA_BY_2_PI * self.seq_config['duration_gy'])
        if rep_idx % 2 == 0: gx = gx_peak
        else:                gx = -gx_peak
        if rep_idx == 0: gy = 0
        else:            gy = (1 / num_phase_encodes) * gy_peak
        seq_repr.loc[(seq_repr['t'] >= self.seq_config['start_time_gx']) & (seq_repr['t'] < (self.seq_config['start_time_gx'] + self.seq_config['duration_gx'])),
                    'gx'] = gx
        seq_repr.loc[(seq_repr['t'] >= self.seq_config['start_time_gy']) & (seq_repr['t'] < (self.seq_config['start_time_gy'] + self.seq_config['duration_gy'])),
                    'gy'] = gy
        
        return seq_repr
    
    def scan(self):
        
        self.reset_kspace_coord()

        num_reps = self.kspace_data.shape[0]
        for rep_idx in tqdm(range(num_reps)):

            # Construct the sequence representation for this rep
            seq_repr = self.construct_seq_repr(rep_idx)

            for i in range(seq_repr['t'].shape[0]):
                
                flip_angle = seq_repr['fa'][i]
                gradient = np.array([seq_repr['gx'][i], seq_repr['gy'][i]])

                # Excitation
                if flip_angle != 0.:
                    self.apply_rf(flip_angle)                

                # PE gradient
                if gradient[1] != 0:
                    self.apply_gradient(gradient)
                
                # FE gradient and readout
                if gradient[0] != 0:
                    self.apply_gradient(gradient, readout=True)

                clock.tick()


class SimClock:
    def __init__(self, time_step):
        self.time_step = time_step
        self.global_time = 0
    def tick(self):
        self.global_time += self.time_step


class FrameRecorder:

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        self.frame_counter = 0

        self.m0 = None
        self.m_state = None
        self.kspace_data = None        
        self.recon = None

        self.kspace_vmin = None
        self.kspace_vmax = None
    
    def set_m_state(self, m_state):
        self.m_state = m_state
    
    def set_kspace_data(self, kspace_data): 
        self.kspace_data = kspace_data
        self.recon = ifft2c(self.kspace_data)
    
    def set_m0(self, m0): 
        self.m0 = m0
        self.kspace_vmin = np.log(np.abs(fft2c(self.m0)) + 1e-2).min()
        self.kspace_vmax = np.log(np.abs(fft2c(self.m0)) + 1e-2).max()

    def save_world_state(self):        

        fig, axs = plt.subplots(2, 3)
        axs[0][0].imshow(self.m_state.real, cmap='gray', vmin=0, vmax=self.m0.max()); axs[0][0].set_title("Spin state (real)")
        axs[1][0].imshow(self.m_state.imag, cmap='gray', vmin=0, vmax=self.m0.max()); axs[1][0].set_title("Spin state (imag)")
        axs[0][1].imshow(np.log(np.abs(self.kspace_data) + 1e-2), cmap='gray', vmin=self.kspace_vmin, vmax=self.kspace_vmax); axs[0][1].set_title("Sampled kspace")
        axs[1][1].imshow(M0, cmap='gray'); axs[1][1].set_title("Ground truth (M0)")
        axs[0][2].imshow(np.abs(self.recon), cmap='gray'); axs[0][2].set_title("Reconstruction (mag)")    
        axs[1][2].imshow(np.angle(self.recon), cmap='gray', vmin=-PI, vmax=PI); axs[1][2].set_title("Reconstruction (phase)")    
        [ax.axis('off') for ax in axs.ravel()]
        fig.tight_layout()
        plt.savefig(f"{self.output_dir}/{str(self.frame_counter).zfill(4)}.png")
        # plt.show()

        self.frame_counter += 1


def fft2c(image):
    image = np.fft.ifftshift(image, axes=(-2, -1))
    kspace = np.fft.fft2(image, axes=(-2, -1))
    kspace = np.fft.fftshift(kspace, axes=(-2, -1))
    return kspace

def ifft2c(kspace):
    kspace = np.fft.ifftshift(kspace, axes=(-2, -1))
    image = np.fft.ifft2(kspace, axes=(-2, -1))
    image = np.fft.fftshift(image, axes=(-2, -1))
    return image


if __name__ == '__main__':

    # Init
    param_maps = {'m0': M0}    
    spin_system = SpinSystem(param_maps, VOXEL_SIZE)
    
    seq_config = {'fa': FA, 'tr': TR, 'te': TE, 
                  'start_time_gx': START_TIME_GX, 'start_time_gy': START_TIME_GY,
                  'duration_gx': DURATION_GX, 'duration_gy': DURATION_GY}
    clock = SimClock(TIME_STEP)
    recorder = FrameRecorder(OUTPUT_DIR)
    recorder.set_m0(M0)
    scanner = Scanner(spin_system, seq_config, clock, recorder)

    # Scan
    scanner.scan()