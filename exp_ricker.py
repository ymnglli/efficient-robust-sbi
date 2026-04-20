from simulators.ricker import ricker
from networks.summary_nets import RickerSummary
from utils.get_nn_models import *
from inference.snpe.snpe_c import SNPE_C as SNPE
from inference.base import *
from utils.torchutils import *
from scipy.stats import qmc
from scipy import stats as stats
from utils.metrics import *
import pickle
import os
import argparse

from utils.timer import Timer

device = torch.device("cpu") # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(args):
    print(device)
    distance = args.distance
    beta = args.beta
    num_simulations = args.num_simulations
    theta_gt = args.theta
    N = args.N
    degree = args.degree
    n_corrupted = int(N * degree)
    n_normal = int(N - n_corrupted)
    prior_mismatch = args.prior_mismatch

    task_name = f"degree={degree}_{distance}_beta={beta}_theta={theta_gt}_num={num_simulations}/{str(args.seed)}"
    root_name = 'objects/NPE/ricker/' + str(task_name)
    timer = Timer(task_name, root_name)

    if not os.path.exists(root_name):
        os.makedirs(root_name)

    timer.start()

    T = 100
    # Use QMC sampling for mmd-efficient
    # TODO check if it is pregenerated already
    if distance == "mmd-efficient":
        sampler = qmc.Sobol(d=T, scramble=True)
        u_uniform = sampler.random(N)
        u = torch.tensor(stats.norm.ppf(u_uniform), dtype=torch.float32, device=device)
    else:
        u = torch.randn(N, T).to(device)

    if prior_mismatch:
        prior = [Uniform(2 * torch.ones(1, device=device), 
                         8 * torch.ones(1, device=device)),
                 torch.distributions.log_normal.LogNormal(
                        loc=torch.tensor([0.5], device=device), 
                        scale=torch.tensor([1], device=device)
                    )]
    else:
        prior = [Uniform(2 * torch.ones(1, device=device), 
                         8 * torch.ones(1, device=device)),
                 Uniform(torch.zeros(1, device=device), 
                         20 * torch.ones(1, device=device))]

    simulator, prior = prepare_for_sbi(ricker(u=u, N=N, T=T), prior)

    sum_net = RickerSummary(input_size=1, hidden_dim=4).to(device)
    neural_posterior = posterior_nn(
        model="maf",
        embedding_net=sum_net,
        hidden_features=20,
        num_transforms=3)

    inference = SNPE(prior=prior, density_estimator=neural_posterior, device=str(device))

    if args.pre_generated_obs:
        if prior_mismatch:
            obs_cont = torch.tensor(np.load("data/ricker_obs_pm.npy")).reshape(-1, 100, 100).to(device)
        else:
            obs_cont = torch.tensor(np.load(f"data/ricker_obs_{int(degree * 10)}.npy"))
    else:
        if prior_mismatch:
            obs_cont = simulator(torch.tensor([[4, 25]])).reshape(-1, N, 100).to(device)
        else:
            theta_gt = torch.tensor(theta_gt)
            theta_cont = torch.tensor([4, 100])
            simulator = ricker(u=u, N=N, T=T)
            obs = simulator(theta_gt).to(device)
            obs_2 = simulator(theta_cont).to(device)
            obs_cont = torch.cat([obs[:n_normal], obs_2[:n_corrupted]], dim=0).reshape(-1, N, 100)

    if args.pre_generated_sim:
        if prior_mismatch:
            theta = torch.tensor(np.load("data/ricker_theta_1000_pm.npy"))
            x = torch.tensor(np.load("data/ricker_x_1000_pm.npy")).reshape(num_simulations, N, 100)
        else:
            theta = torch.tensor(np.load("data/ricker_theta_1000.npy"))
            x = torch.tensor(np.load("data/ricker_x_1000.npy")).reshape(num_simulations, N, 100)
    else:
        theta, x = simulate_for_sbi(simulator, prior, num_simulations=num_simulations)

    timer.lap()
    x = x.reshape(num_simulations, N, 100).to(device)
    theta = theta.to(device)
    density_estimator = inference.append_simulations(theta, x.unsqueeze(1)).train(
        distance=distance, 
        x_obs=obs_cont, 
        beta=beta)

    # increase the prior range in case we can't generate thetas for mis-specified observation
    prior_new = [Uniform(2 * torch.ones(1), 8 * torch.ones(1)),
                 Uniform(torch.zeros(1), 80 * torch.ones(1))]
    simulator, prior_new = prepare_for_sbi(ricker(u=u, N=N, T=T), prior_new)
    posterior = inference.build_posterior(density_estimator, prior=prior_new)

    with open(root_name + "/posterior.pkl", "wb") as handle:
        pickle.dump(posterior, handle)

    torch.save(sum_net, root_name + "/sum_net.pkl")
    torch.save(density_estimator, root_name + "/density_estimator.pkl")

    if args.keep_inference:
        with open(root_name + "/inference.pkl", "wb") as handle:
            pickle.dump(inference, handle)
    timer.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", type=float, default=1.0, help="regularization weight")
    parser.add_argument("--degree", type=float, default=0.2, help="degree of mis-specification")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--distance", type=str, default="mmd", choices=["euclidean", "none", "mmd", "mmd-efficient"])
    parser.add_argument("--num_simulations", type=int, default=1000, help="number of simulations")
    parser.add_argument("--theta", type=list, default=[4, 10], help="ground truth theta")
    parser.add_argument("--N", type=int, default=100, help="Number of realizations for each set of theta")
    parser.add_argument("--prior-mismatch", action="store_true", help="whether use mis-specified prior")
    parser.add_argument("--pre-generated-sim", action="store_true", help="generate simulation data online or not")
    parser.add_argument("--pre-generated-obs", action="store_true", help="generate observation data online or not")
    parser.add_argument("--keep-inference", action="store_true", help="save inference model or not")
    args = parser.parse_args()
    main(args)
