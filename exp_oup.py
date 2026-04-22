import torch

from simulators.oup import oup

from networks.summary_nets import OUPSummary
from utils.get_nn_models import *
from inference.snpe.snpe_c import SNPE_C as SNPE
from inference.base import *
from utils.torchutils import *
from utils.user_input_checks import process_prior
import pickle
import os
import argparse

device = torch.device("cpu") #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Note: Run generate_data.py to generate data first

def main(args):
    var = args.var
    distance = args.distance
    beta = args.beta
    num_simulations = args.num_simulations
    theta_gt = args.theta
    N = args.N
    degree = args.degree
    n_corrupted = int(N * degree)
    n_normal = int(N - n_corrupted)
    sampling = args.sampling
    sample_size = args.sample_size

    task_name = f"degree={degree}_var={var}_{distance}_beta={beta}_theta={theta_gt}_num={num_simulations}_size={sample_size}/{str(args.seed)}"
    root_name = 'objects/oup_final/' + str(task_name)
    if not os.path.exists(root_name):
        os.makedirs(root_name)

    prior = [Uniform(torch.zeros(1).to(device), 2 * torch.ones(1).to(device)),
             Uniform(-2 * torch.ones(1).to(device), 2 * torch.ones(1).to(device))]

    prior, _, _ = process_prior(prior)

    sum_net = OUPSummary(input_size=1, hidden_dim=2, N=N).to(device)
    neural_posterior = posterior_nn(
        model="maf",
        embedding_net=sum_net,
        hidden_features=20,
        num_transforms=3)

    inference = SNPE(prior=prior, density_estimator=neural_posterior, device=str(device))

    if args.pre_generated_obs:
        obs_cont = torch.tensor(np.load(f"data/oup_obs_{int(degree * 10)}.npy"))
    else:
        raise RuntimeError("This pipeline requires pre-generated observations")

    suffix = "_qmc" if sampling == "qmc" else ""
    if args.pre_generated_sim:
        theta = torch.tensor(np.load(f"data/oup_theta_{num_simulations}{suffix}.npy"))
        x = torch.tensor(np.load(f"data/oup_x_{num_simulations}{suffix}.npy")).reshape(num_simulations, N, 25)
        u = torch.tensor(np.load(f"data/oup_u_{num_simulations}{suffix}.npy")).reshape(num_simulations, N, 25)
    else:
        raise RuntimeError("This pipeline requires pre-generated simulations")

    x = x.reshape(num_simulations, N, 25).to(device)
    theta = theta.to(device)
    u = u.to(device)
    density_estimator = inference.append_simulations(theta, x.unsqueeze(1), u).train(
        distance=distance, 
        x_obs=obs_cont, 
        beta=beta, 
        sample_size=sample_size,
        training_batch_size=64)

    # increase the prior range in case we can't generate thetas for mis-specified observation
    prior_new = [Uniform(-20 * torch.ones(1), 20 * torch.ones(1)),
                 Uniform(-20 * torch.ones(1), 20 * torch.ones(1))]
    prior_new, _, _ = process_prior(prior_new)
    posterior = inference.build_posterior(density_estimator, prior=prior_new)

    with open(root_name + "/posterior.pkl", "wb") as handle:
        pickle.dump(posterior, handle)

    torch.save(sum_net, root_name + "/sum_net.pkl")
    torch.save(density_estimator, root_name + "/density_estimator.pkl")

    if args.keep_inference:
        with open(root_name + "/inference.pkl", "wb") as handle:
            pickle.dump(inference, handle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--var", type=float, default=0.1, help="variance of the OU process")
    parser.add_argument("--beta", type=float, default=1.0, help="regularization weight")
    parser.add_argument("--degree", type=float, default=0.2, help="degree of mis-specification")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--distance", type=str, default="mmd", choices=["euclidean", "none", "mmd", "mmd-efficient"])
    parser.add_argument("--num_simulations", type=int, default=1024, help="number of simulations")
    parser.add_argument("--theta", type=list, default=[0.5, 1.0], help="ground truth theta")
    parser.add_argument("--N", type=int, default=128, help="Number of realizations for each set of theta")
    parser.add_argument("--pre-generated-sim", action="store_true", help="generate simulation data online or not")
    parser.add_argument("--pre-generated-obs", action="store_true", help="generate observation data online or not")
    parser.add_argument("--keep-inference", action="store_true", help="save inference model or not")
    parser.add_argument("--sampling", type=str, default="mc", choices=["mc", "qmc"], help="sampling method to use")
    parser.add_argument("--sample-size", type=int, default=256, help="sample size used to estimate MMD in simulated and observed data")
    args = parser.parse_args()
    main(args)
