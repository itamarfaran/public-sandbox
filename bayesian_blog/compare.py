import numpy as np
from scipy.stats import beta, norm

from main import \
    alpha_a, beta_a, alpha_b, beta_b, \
    mu_a, sd_a, mu_b, sd_b, \
    d1_beta, d2_beta, d1_norm, d2_norm, \
    risk_beta, risk_norm


def print_approx_and_sim(approx, sim, tag=None):
    if tag:
        print(tag)
    print(f'approximation: {approx}')
    print(f'simulations: {sim}')
    print(f'difference: {np.abs(approx - sim)}')
    print(f'relative difference: {np.abs(approx / sim - 1)}')
    print('')


N_SIM = 500000
RANDOM_STATE = (1235, 3418, 8731, 6754)

rate_a_sim = beta.rvs(alpha_a, beta_a, size=N_SIM, random_state=RANDOM_STATE[0])
rate_b_sim = beta.rvs(alpha_b, beta_b, size=N_SIM, random_state=RANDOM_STATE[1])

mean_a_sim = norm.rvs(mu_a, sd_a, size=N_SIM, random_state=RANDOM_STATE[2])
mean_b_sim = norm.rvs(mu_b, sd_b, size=N_SIM, random_state=RANDOM_STATE[3])


if __name__ == '__main__':
    par = 0
    print_approx_and_sim(d1_beta.sf(par),
                         np.mean(rate_b_sim - rate_a_sim > par),
                         f'beta-binomial probability difference is greater than {par}')

    ccr = .05  # complementary of confidence rate
    quantiles = ccr / 2, 1 - ccr / 2
    print_approx_and_sim(np.exp(d2_beta.ppf(quantiles)) - 1,
                         np.quantile(rate_b_sim / rate_a_sim - 1, quantiles),
                         f'beta-binomial credibility interval of {1 - ccr}')

    par = .98
    print_approx_and_sim(d2_norm.cdf(np.log(par)),
                         np.mean(mean_b_sim / mean_a_sim <= par),
                         f'normal-normal probability uplift is lower than {par - 1}')

    ccr = .1
    quantiles = ccr / 2, 1 - ccr / 2
    print_approx_and_sim(d1_norm.ppf(quantiles),
                         np.quantile(mean_b_sim - mean_a_sim, quantiles),
                         f'normal-normal credibility interval of {1 - ccr}')

    print_approx_and_sim(risk_beta[0],
                         np.mean((rate_b_sim - rate_a_sim) * (rate_b_sim > rate_a_sim)),
                         'beta-binomial risk of a')

    print_approx_and_sim(risk_beta[1],
                         np.mean((rate_a_sim - rate_b_sim) * (rate_a_sim > rate_b_sim)),
                         'beta-binomial risk of b')

    print_approx_and_sim(risk_norm[0],
                         np.mean((mean_b_sim - mean_a_sim) * (mean_b_sim > mean_a_sim)),
                         'normal-normal risk of a')

    print_approx_and_sim(risk_norm[1],
                         np.mean((mean_a_sim - mean_b_sim) * (mean_a_sim > mean_b_sim)),
                         'normal-normal risk of b')
