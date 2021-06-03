import numpy as np
from scipy.stats import beta, binom, expon, gamma, norm


RANDOM_STATE = (3514, 4348, 1535, 9753, 5384)
N_SIM = 500000
SAMPLE_SIZE = 2000


def print_approx_and_sim(norm_, prod_, tag=None):
    if tag:
        print(tag)
    print(f'gaussian model: {norm_}')
    print(f'beta-exponential model: {prod_}')
    print(f'difference: {np.abs(norm_ - prod_)}')
    print(f'relative difference: {np.abs(norm_ / prod_ - 1)}')
    print('')


beta_prior = dict(a=20, b=80)
gamma_prior = dict(a=5, scale=20)

theta_ = beta.rvs(**beta_prior, random_state=RANDOM_STATE[0])
lambda_ = gamma.rvs(**gamma_prior, random_state=RANDOM_STATE[1])

mu_prior = beta.mean(**beta_prior) * gamma.mean(**gamma_prior)
sigma_sqrd_prior = beta.moment(2, **beta_prior) * gamma.moment(2, **gamma_prior) - np.power(mu_prior, 2)


u = binom.rvs(1, theta_, size=SAMPLE_SIZE)
v = expon.rvs(scale=lambda_, size=SAMPLE_SIZE)
x = u * v

beta_posterior = dict(a=beta_prior['a'] + u.sum(), b=beta_prior['b'] + SAMPLE_SIZE - u.sum())
gamma_posterior = dict(a=gamma_prior['a'] + u.sum(), scale=1 / (1 / gamma_prior['scale'] + x.sum()))

inv_vars = 1 / sigma_sqrd_prior, SAMPLE_SIZE / x.var()
mu_posterior = dict(loc=np.average((mu_prior, x.mean()), weights=inv_vars),
                    scale=1 / np.sqrt(np.sum(inv_vars)))


prod_sim = beta.rvs(**beta_posterior, size=N_SIM, random_state=RANDOM_STATE[2]) / \
           gamma.rvs(**gamma_posterior, size=N_SIM, random_state=RANDOM_STATE[3])
norm_sim = norm.rvs(**mu_posterior, size=N_SIM, random_state=RANDOM_STATE[4])


if __name__ == '__main__':
    quantiles = [.1, .2, .5, .8, .9]
    print_approx_and_sim(prod_sim.mean(), norm_sim.mean(), 'mean')
    print_approx_and_sim(prod_sim.std(), norm_sim.std(), 'std')
    print_approx_and_sim(np.quantile(prod_sim, quantiles), np.quantile(norm_sim, quantiles), 'quantiles')
