import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    try:
        plt.style.use('seaborn-v0_8-paper')
    except:
        try:
            plt.style.use('seaborn-paper')
        except:
            pass
    sns.set_palette(['#1f77b4', '#d62728', '#008080'])
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times', 'Times New Roman', 'DejaVu Serif'],
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.dpi': 300
    })

    d = np.linspace(0.5, 50, 500)
    I_0 = 100
    
    # Mathematical models
    I_raw = I_0 / (d**2)
    eta = 1 - np.exp(-0.8 * d)
    I_sensor = eta * I_raw
    R_comp = I_sensor * (d**2)
    
    # Add simulated noise to show noise expansion at long ranges
    np.random.seed(42)
    noise_baseline = 0.5
    sensor_noise = np.random.normal(0, noise_baseline, size=len(d))
    I_sensor_noisy = np.clip(I_sensor + sensor_noise, 0, None)
    R_comp_noisy = I_sensor_noisy * (d**2)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    # Top Plot: Raw vs Sensor Intensity
    ax1.plot(d, I_raw, label=r'Theoretical $I(d) = I_0 / d^2$', linestyle='--', color='gray')
    ax1.plot(d, I_sensor, label=r'Sensor $I_{sensor}(d)$ (with defocus)', color='#1f77b4', linewidth=2)
    ax1.fill_between(d, 0, ax1.get_ylim()[1], where=(d < 5), color='#d62728', alpha=0.15, label='Near-Range Defocusing (<5m)')
    ax1.set_ylabel('Intensity $I$')
    ax1.set_title('Raw vs. Sensor Intensity Decay')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 150)

    # Bottom Plot: Compensated Reflectance
    ax2.plot(d, R_comp_noisy, label=r'Compensated $R_{comp}$ (with noise)', color='#008080', linewidth=1.5, alpha=0.8)
    ax2.plot(d, R_comp, label=r'Ideal Compensated $R_{comp}$', color='#1f77b4', linewidth=2)
    ax2.fill_between(d, 0, ax2.get_ylim()[1], where=(d > 30), color='#9467bd', alpha=0.15, label='Noise Amplification (>30m)')
    ax2.fill_between(d, 0, ax2.get_ylim()[1], where=(d < 5), color='#d62728', alpha=0.15)
    
    ax2.set_xlabel('Distance $d$ (meters)')
    ax2.set_ylabel('Reflectance $R_{comp}$')
    ax2.set_title('Distance-Compensated Reflectance')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 200)

    plt.tight_layout()
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(output_dir, 'fig1_3_lidar_distance_decay.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'fig1_3_lidar_distance_decay.pdf'))
    plt.close()

if __name__ == '__main__':
    main()
