import os
import torch
import numpy as np
from inference.base import *
from simulators import *
from utils.torchutils import *
from torch.distributions import Uniform, LogNormal
from scipy.stats import qmc
from scipy import stats as stats

DEVICE = torch.device("cpu")
DATA_DIR = "data"
DEGREES = [0, 0.1, 0.2]
NUM_SIMULATIONS = 1000
# N_SAMPLES should be a power of 2 for Sobol sampling
N_SAMPLES = 128
T_RICKER = 100
T_OUP = 25

def save_numpy(path, tensor):
    """Helper to handle detaching and conversion."""
    np.save(path, tensor.detach().cpu().numpy())

def generate_u(N, timesteps, engine=None):
    """
    Generates a (N, T) block of noise.
    If engine is provided, it draws the next sequence in the Sobol chain.
    """
    if engine is not None:
        u_uniform = engine.random(N)
        # Map [0, 1] to [0.0001, 0.9999]
        # This keeps Gaussian noise within approx +/- 3.7 standard deviations
        u_safe = 1e-4 + (u_uniform * (1 - 2e-4))
        return torch.tensor(stats.norm.ppf(u_safe), dtype=torch.float32, device=DEVICE)
    else:
        return torch.randn(N, timesteps).to(DEVICE)

def generate_contaminated_obs(theta_gt, theta_cont, n_total, time_steps, model_name):
    """Generates ground-truth/contaminated observations using standard MC noise."""
    u_fixed = generate_u(n_total, time_steps, engine=None) 
    sim = ricker(u=u_fixed, N=n_total, T=time_steps)
    
    for degree in DEGREES:
        n_corrupted = int(n_total * degree)
        n_normal = n_total - n_corrupted
        
        obs_clean = sim(theta_gt).to(DEVICE)
        obs_dirty = sim(theta_cont).to(DEVICE)
        
        combined = torch.cat([obs_clean[:n_normal], obs_dirty[:n_corrupted]], dim=0)
        combined = combined.reshape(-1, n_total, time_steps)
        
        suffix = int(degree * 10)
        save_numpy(f"{DATA_DIR}/{model_name}_obs_{suffix}.npy", combined)

def run_sbi_simulation(prior_list, model_name, time_steps, use_qmc=False):
    """
    Handles the full simulation loop for a given prior/model.
    Initializes the Sobol engine ONCE per dataset to maintain sequence integrity.
    """
    noise_logs = []
    
    # Initialize the Sobol engine once here. 
    # d=time_steps because each 'sample' is a time series of length T.
    engine = qmc.Sobol(d=time_steps, scramble=True) if use_qmc else None

    def simulator_wrapper(theta):
        # Pulls the next N_SAMPLES from the persistent engine.
        u = generate_u(N_SAMPLES, time_steps, engine=engine)
        noise_logs.append(u)
        if model_name == "oup":
            sim_inst = oup(u=u, N=N_SAMPLES, n=time_steps)
        elif model_name == "ricker": 
            sim_inst = ricker(u=u, N=N_SAMPLES, T=time_steps)
        else:
            raise RuntimeError(f"{model_name} not supported in this pipeline")
        return sim_inst(theta)

    sim, prior = prepare_for_sbi(simulator_wrapper, prior_list)
    noise_logs.clear()

    theta, x = simulate_for_sbi(sim, prior, NUM_SIMULATIONS)
    file_suffix = "_qmc" if use_qmc else ""

    # Stack logs to create (NUM_SIMULATIONS, N_SAMPLES, T)
    u_tensor = torch.stack(noise_logs)
    
    theta_reshaped = theta.reshape(NUM_SIMULATIONS, -1)
    x_reshaped = x.reshape(NUM_SIMULATIONS, N_SAMPLES, time_steps)
    
    save_numpy(f"{DATA_DIR}/{model_name}_theta_{NUM_SIMULATIONS}{file_suffix}.npy", theta_reshaped)
    save_numpy(f"{DATA_DIR}/{model_name}_x_{NUM_SIMULATIONS}{file_suffix}.npy", x_reshaped)
    save_numpy(f"{DATA_DIR}/{model_name}_u_{NUM_SIMULATIONS}{file_suffix}.npy", u_tensor)

if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    generate_contaminated_obs(torch.tensor([4, 10]), torch.tensor([4, 100]), 
                              N_SAMPLES, T_RICKER, "ricker")

    u_pm = generate_u(N_SAMPLES, T_RICKER, engine=None)
    sim_pm_obs = ricker(u=u_pm, N=N_SAMPLES, T=T_RICKER)
    obs_pm = sim_pm_obs(torch.tensor([[4, 25]])).reshape(-1, N_SAMPLES, T_RICKER)
    save_numpy(f"{DATA_DIR}/ricker_obs_pm.npy", obs_pm)

    prior_ricker = [Uniform(2 * torch.ones(1), 8 * torch.ones(1)),
                    Uniform(torch.zeros(1), 20 * torch.ones(1))]

    prior_ricker_pm = [
        Uniform(2 * torch.ones(1), 8 * torch.ones(1)),
        LogNormal(loc=torch.tensor([0.5]), scale=torch.tensor([1.0]))
    ]

    run_sbi_simulation(prior_ricker, "ricker", T_RICKER, use_qmc=False)
    run_sbi_simulation(prior_ricker, "ricker", T_RICKER, use_qmc=True)
    run_sbi_simulation(prior_ricker_pm, "ricker_pm", T_RICKER, use_qmc=False)
    run_sbi_simulation(prior_ricker_pm, "ricker_pm", T_RICKER, use_qmc=True)
    
    # OUP
    prior_oup = [
        Uniform(torch.zeros(1).to(DEVICE), 2 * torch.ones(1).to(DEVICE)),
        Uniform(-2 * torch.ones(1).to(DEVICE), 2 * torch.ones(1).to(DEVICE))
    ]
    
    generate_contaminated_obs(torch.tensor([0.5, 1.0]), torch.tensor([-0.5, 1]), 
                              N_SAMPLES, T_OUP, "oup")
    run_sbi_simulation(prior_oup, "oup", T_OUP, use_qmc=False)
    run_sbi_simulation(prior_oup, "oup", T_OUP, use_qmc=True)
    
    """ Original script was broken here
    turin = turin(B=4e9, Ns=801, N=N, tau0=0)
    prior_turin = [Uniform(1e-9*torch.ones(1).to(device), 1e-8*torch.ones(1).to(device)),
                   Uniform(1e-9*torch.ones(1).to(device), 1e-8*torch.ones(1).to(device)),
                   Uniform(1e7*torch.ones(1).to(device), 5e9*torch.ones(1).to(device)),
                   Uniform(1e-10*torch.ones(1).to(device), 1e-9*torch.ones(1).to(device))]
    theta_gt = torch.tensor([10**(-8.4), 7.8e-9, 1e9, 2.8e-10])

    theta, x = simulate_data(turin, prior_turin, num_simulations)
    np.save(f"{data_dir}/turin_theta_{num_simulations}.npy", theta.reshape(num_simulations, 4).detach().numpy())
    np.save(f"{data_dir}/turin_x_{num_simulations}.npy", x.reshape(num_simulations, N, 801).detach().numpy())
    """