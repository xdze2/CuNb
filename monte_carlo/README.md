# Monte Carlo Methods



## 1. Pourquoi faire ? Exploration de l'espace des paramètres

A function $f$ of a parameter set : $\theta = \left\{\theta_i, \, ...\right\}$  

- Optimisation:
$$
\mathop{argmin}_{\theta}\;f(\theta)
$$


- Integration: $\int_{\theta \in \Omega} f(\theta)d\theta$  ou $\int_{\theta \in \Omega} f(\theta) g(\theta)d\theta / \int g(\theta) d\theta$ 

- Minimum path finding, reaction constant... ?
$$
\mathop{argmin}_{\Gamma} \int_\Gamma f(x)\, dx
$$

* Parameter estimation: $p(\theta | Y)$ ... $\theta \pm \Delta\theta$ ... 
  function approximation / fit:

* $$
  \mathop{argmin}_{\mu} \int ||g_\mu(x)-f(x) ||^2 \,dx
  $$



i.e. function -probability density function- estimation... generate random samples with a given probabilty density function

Sondage: ex. average from N random samples 

## Pourquoi comme ça ?

​   plutôt que des methodes de maillage... grand nombre de paramètres



## Sampling:

- Uniform sampling, Random sampling:  
  no use of knowledge about the function, the sampling is not optimised as many samples are where the value of the function is not significant

- Importance sampling: $f(x)/w(x)$... but $w(x)$ is not known

- Metropolis sampling: marche aléatoire avec rejet des pas hors du domain... seul une mesure relative est obtenue
  -simulated annealing: variable T-

- Iterative construction of $w(x)$:
  cf. Vega algorithm (parameter space partitioning)
  [Mutated Kd-tree importance sampling](https://www.researchgate.net/publication/228923004_Mutated_Kd-tree_importance_sampling)

- ​

  ​

marcheur aléatoire, deterministic (Newton, CG, simplex...)  
... use of derivative (variational), evolutionary/genetic algo., simulated annealing

then, a relative measure is obtained 
$$
\int_{\theta \in \Omega} f(\theta)d\theta  \rightarrow  \int_{\theta \in \Omega} f(\theta)d\theta
$$


  adaptative/importance sampling, refinement


### Monte Carlo

- Gibbs sampling: one $\theta_i$ at a time
- Metropolis-Hasting: jumping condition (rejecting sampling)
- Hamiltonian or Hybrid Monte Carlo: uses gradient



Laplace's method: posterior distribution is assumed to be Gaussian... how we sample? iterative importance sampling... ?



metric in the parameter space... volume = chou x carotte... renormalization/adimensionement





##  Liens

- https://interstices.info/le-jeu-de-go-et-la-revolution-de-monte-carlo/