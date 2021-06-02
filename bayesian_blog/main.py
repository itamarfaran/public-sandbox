import numpy as np
from scipy.stats import norm, beta
from scipy.special import digamma, polygamma, roots_hermitenorm  # , roots_sh_jacobi
from orthogonal import roots_sh_jacobi


# In[parameters definition]

x_a, n_a = 254, 1283  # converted & total users in A
x_b, n_b = 289, 1321  # converted & total users in B

m_a, s_a = 52.3, 14.1  # average & sd over users in A
m_b, s_b = 52.8, 13.7  # average & sd over users in B

alpha_0, beta_0 = 1, 1  # Beta prior
mu0, s0, n0 = 0, 1, 1  # Gaussian prior


# In[update priors]

# updating Beta prior
alpha_a = alpha_0 + x_a
beta_a = beta_0 + n_a - x_a

alpha_b = alpha_0 + x_b
beta_b = beta_0 + n_b - x_b


# updating Gaussian prior is a bit more tricky
inv_sds = n0 / np.power(s0, 2), n_a / np.power(s_a, 2)
mu_a = np.average((mu0, m_a), weights=inv_sds)
sd_a = 1 / np.sqrt(np.sum(inv_sds))

inv_sds = n0 / np.power(s0, 2), n_b / np.power(s_b, 2)
mu_b = np.average((mu0, m_b), weights=inv_sds)

sd_b = 1 / np.sqrt(np.sum(inv_sds))


# In[calculate "easy" metrics]

log_beta_mean = lambda a, b: digamma(a) - digamma(a + b)
var_beta_mean = lambda a, b: polygamma(1, a) - polygamma(1, a + b)


d1_beta = norm(loc=beta.mean(alpha_b, beta_b) - beta.mean(alpha_a, beta_a),
               scale=np.sqrt(beta.var(alpha_b, beta_b) + beta.var(alpha_a, beta_a)))
d2_beta = norm(loc=log_beta_mean(alpha_b, beta_b) - log_beta_mean(alpha_a, beta_a),
               scale=np.sqrt(var_beta_mean(alpha_b, beta_b) + var_beta_mean(alpha_a, beta_a)))

d1_norm = norm(loc=mu_b - mu_a, scale=np.sqrt(sd_a ** 2 + sd_b ** 2))
d2_norm = norm(loc=np.log(mu_b) - np.log(mu_a), scale=np.sqrt((sd_a / mu_a)**2 + (sd_b / mu_b)**2))


# In[calculate risk with gq]

# The following throws an integer overflow error when a + b are too large
# Use the log trick instead (see my PR at scipy)
def beta_gq(n, a, b):
    x, w, m = roots_sh_jacobi(n, a + b - 1, a, True)
    w /= m
    return x, w


nodes_a, weights_a = beta_gq(24, alpha_a, beta_a)
nodes_b, weights_b = beta_gq(24, alpha_b, beta_b)

gq = sum(nodes_a * beta.cdf(nodes_a, alpha_b, beta_b) * weights_a) + \
     sum(nodes_b * beta.cdf(nodes_b, alpha_a, beta_a) * weights_b)
risk_beta = gq - beta.mean((alpha_a, alpha_b), (beta_a, beta_b))


def norm_gq(n, loc, scale):
    x, w, m = roots_hermitenorm(n, True)
    x = scale * x + loc
    w /= m
    return x, w


nodes_a, weights_a = norm_gq(24, mu_a, sd_a)
nodes_b, weights_b = norm_gq(24, mu_b, sd_b)

gq = sum(nodes_a * norm.cdf(nodes_a, mu_b, sd_b) * weights_a) + \
     sum(nodes_b * norm.cdf(nodes_b, mu_a, sd_a) * weights_b)
risk_norm = gq - norm.mean((mu_a, mu_b))


if __name__ == '__main__':
    print(f'The probability the conversion in B is higher is {d1_beta.sf(0)}')
    print(f'The 95% crediblity interval of (p_b/p_a-1) is {np.exp(d2_beta.ppf((.025, .975))) - 1}')
    # >>> The probability the conversion in B is higher is 0.9040503042127781
    # >>> The 95% crediblity interval of (p_b/p_a-1) is [-0.0489547   0.28350359]

    print(f'The probability the average income in B is 2% lower (or worse) is {d2_norm.cdf(np.log(.98))}')
    print(f'The 95% crediblity interval of (m_b - m_a) is {d1_norm.ppf((.025, .975))}')
    # >>> The probability the average income in B is 2% lower (or worse) is 0.00011622384023304196
    # >>> The 95% crediblity interval of (m_b - m_a) is [-0.04834271  1.94494332]

    print(f'The risk of choosing A is losing {risk_beta[0]} conversions per user.\n'
          f'The risk of choosing B is losing {risk_beta[1]} conversions per user.')
    # >>> The risk of choosing A is losing 0.021472801833822552 conversions per user.
    # >>> The risk of choosing B is losing 0.0007175909729974506 conversions per user.

    print(f'The risk of choosing A is losing {risk_norm[0]}$ per user.\n'
          f'The risk of choosing B is losing {risk_norm[1]}$ per user.')
    # >>> The risk of choosing A is losing 0.9544550905343314$ per user.
    # >>> The risk of choosing B is losing 0.006154785995697409$ per user.
