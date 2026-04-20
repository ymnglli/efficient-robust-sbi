import torch
import numpy as np


class ricker():
    # Workaround to access u with the sbi library
    # Do not use batch size > 1 because u will be reused
    def __init__(self, u, N=20, T=100):
        self.N = N
        self.u = u
        self.T = T

    def __call__(self, theta, *args, **kwargs):
        if len(theta.shape) == 1:
            logr = theta[0]
            phi = theta[1]
        else:
            logr = theta[0, 0]
            phi = theta[0, 1]
        if phi < 0:
            phi = 0.001
        sigma = 0.3

        N0 = 1
        nSamples = self.N

        Y = torch.zeros(size=(nSamples, self.T))

        for i in range(nSamples):
            et = self.u[i]
            Nt = torch.zeros(size=(self.T + 1,))

            Nt[0] = N0

            for t in range(1, self.T + 1):
                # Calculate the log-population first to prevent overflow
                # Add 1e-10 to log to prevent log(0) -> -inf
                log_next_Nt = logr + torch.log(Nt[t - 1] + 1e-10) - Nt[t - 1] + sigma * et[t - 1]
                
                # Clamp the exponent to prevent exp(700+) -> inf
                # 50 is safe for float32/float64
                log_next_Nt = torch.clamp(log_next_Nt, max=50) 
                
                Nt[t] = torch.exp(log_next_Nt)
                
                rate = phi * Nt[t]
                rate = torch.clamp(rate, min=0.0)
                
                if torch.isnan(rate) or rate <= 0:
                    rate = torch.tensor(1e-5)
                    
                Y[i, t - 1] = torch.poisson(rate)
        return Y

    def get_name(self):
        return "ricker"
