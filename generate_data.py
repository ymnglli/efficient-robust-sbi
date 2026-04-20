import os
from inference.base import *
from simulators import *
from utils.torchutils import *
from torch.distributions import Uniform, LogNormal

device = torch.device("cpu")

def simulate_data(simulator, prior, num_simulations):
    theta, x = simulate_for_sbi(simulator, prior, num_simulations)
    return theta, x


if __name__ == "__main__":
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    degrees = [0, 0.1, 0.2]
    N = 100
    num_simulations = 1000
    
    ricker_inst = ricker(N=N)
    prior_ricker = [Uniform(2 * torch.ones(1), 8 * torch.ones(1)),
                    Uniform(torch.zeros(1), 20 * torch.ones(1))]
    theta_gt = torch.tensor([4, 10])
    theta_cont = torch.tensor([4, 100])

    for degree in degrees:
        n_corrupted = int(N * degree)
        n_normal = int(N - n_corrupted)
        obs = ricker_inst(theta_gt).to(device)
        obs_2 = ricker_inst(theta_cont).to(device)
        obs_cont = torch.cat([obs[:n_normal], obs_2[:n_corrupted]], dim=0).reshape(-1, N, 100)
        np.save(f"{data_dir}/ricker_obs_{int(degree * 10)}.npy", obs_cont.detach().numpy())

    ricker, prior_ricker = prepare_for_sbi(ricker_inst, prior_ricker)
    theta, x = simulate_data(ricker, prior_ricker, num_simulations)
    np.save(f"{data_dir}/ricker_theta_{num_simulations}.npy", theta.reshape(num_simulations, 2).detach().numpy())
    np.save(f"{data_dir}/ricker_x_{num_simulations}.npy", x.reshape(num_simulations, N, 100).detach().numpy())

    prior_ricker_pm = [
        Uniform(2 * torch.ones(1), 8 * torch.ones(1)),
        LogNormal(loc=torch.tensor([0.5]), scale=torch.tensor([1.0]))
    ]

    ricker, prior_ricker_pm = prepare_for_sbi(ricker_inst, prior_ricker_pm)
    obs_cont_pm = ricker(torch.tensor([[4, 25]])).reshape(-1, N, 100).to(device)
    np.save(f"{data_dir}/ricker_obs_pm.npy", obs_cont_pm.detach().numpy())
    theta_pm, x_pm = simulate_data(ricker, prior_ricker_pm, num_simulations)
    np.save(f"{data_dir}/ricker_theta_{num_simulations}_pm.npy", theta_pm.reshape(num_simulations, 2).detach().numpy())
    np.save(f"{data_dir}/ricker_x_{num_simulations}_pm.npy", x_pm.reshape(num_simulations, N, 100).detach().numpy())

    oup = oup(N=N)
    prior_oup = [Uniform(torch.zeros(1).to(device), 2 * torch.ones(1).to(device)),
                 Uniform(-2 * torch.ones(1).to(device), 2 * torch.ones(1).to(device))]
    theta_gt = torch.tensor([0.5, 1.0])
    theta_cont = torch.tensor([-0.5, 1])

    for degree in degrees:
        n_corrupted = int(N * degree)
        n_normal = int(N - n_corrupted)
        obs = oup(theta_gt).to(device)
        obs_2 = oup(theta_cont).to(device)
        obs_cont = torch.cat([obs[:n_normal], obs_2[:n_corrupted]], dim=0).reshape(-1, N, 25)
        np.save(f"{data_dir}/oup_obs_{int(degree * 10)}.npy", obs_cont.detach().numpy())

    oup, prior_oup = prepare_for_sbi(oup, prior_oup)
    theta, x = simulate_data(oup, prior_oup, num_simulations)
    np.save(f"{data_dir}/oup_theta_{num_simulations}.npy", theta.reshape(num_simulations, 2).detach().numpy())
    np.save(f"{data_dir}/oup_x_{num_simulations}.npy", x.reshape(num_simulations, N, 25).detach().numpy())

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