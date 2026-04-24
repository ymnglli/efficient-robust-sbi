import os
from exp_ricker import main

class Arg:
    def __init__(self, beta, degree, seed, distance, sampling, sample_size):
        self.beta=beta
        self.degree=degree
        self.seed=seed
        self.distance=distance
        self.num_simulations=1024
        self.theta=[4,10]
        self.N=128
        self.prior_mismatch = False
        self.pre_generated_sim=True
        self.pre_generated_obs=True
        self.keep_inference=True
        self.sampling=sampling
        self.sample_size=sample_size

def runner():
    degrees = [0.0, 0.1, 0.2]
    seeds = range(5)
    distances = ["mmd", "mmd-efficient"]
    sample_sizes = [32, 128, 256]
    beta = 2.0
    
    for seed in seeds:
        for degree in degrees:
            for sample_size in sample_sizes:
                for distance in distances:

                    sampling = "qmc" if distance == "mmd-efficient" else "mc"

                    modelPath = f"objects/NPE/ricker/degree={degree}_{distance}_beta={beta}_theta=[4, 10]_num=1024_size={sample_size}/{seed}"

                    inference = os.path.join(modelPath, "inference.pkl")

                    print(modelPath)

                    if (os.path.exists(inference)):
                        print("Inference file exists: model was already trained, skipping")
                        continue

                    exp_args = Arg(
                        beta=beta,
                        degree=degree,
                        seed=seed,
                        distance=distance,
                        sampling=sampling,
                        sample_size=sample_size
                    )

                    main(exp_args)

if (__name__ == "__main__"):
    runner()