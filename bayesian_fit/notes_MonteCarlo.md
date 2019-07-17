## Why Monte Carlo



## Exploration de l'espace des paramètres:

- optimisation:
$$
\mathop{argmin}_{\theta}\;f(\theta)
$$


- integration: $\int_{\theta \in \Omega} f(\theta)d\theta$

- minimum path finding, reaction constant... ?
$$
\mathop{argmin}_{\Gamma} \int_\Gamma f(x)\, dx
$$

* Parameter estimation: $p(\theta | Y)$ ... $\theta \pm \Delta\theta$ ... 
  function approximation:

* $$
  \mathop{argmin}_{\mu} \int ||g_\mu(x)-f(x) ||^2 \,dx
  $$

  ​



## Sampling:

- brute force:
  uniform sampling, random sampling --> no use of knowledge about the function, the sampling is not optimised

- iterative:
  marcheur aléatoire, deterministic (Newton, CG, simplex...)  
  ... use of derivative (variational), evolutionary/genetic algo., simulated annealing


  adaptative/importance sampling, refinement


### Monte Carlo

- Gibbs sampling: one $\theta_i$ at a time
- Metropolis-Hasting: jumping condition (rejecting sampling)
- Hamiltonian or Hybrid Monte Carlo: uses gradient



Laplace's method: posterior distribution is assumed to be Gaussian... how we sample? iterative importance sampling... ?



metric in the parameter space... volume = chou x carotte... renormalization/adimensionement