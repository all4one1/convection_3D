# Fixed heat flux

Mean scalar value on the bottom boundary:

$$
\bar{\phi}_{\mathrm{bottom}}
= \frac{1}{N}
  \sum_{k=1}^{n_z-1}
  \sum_{i=1}^{n_x-1} \phi_{i,0,k}
$$

Mean scalar value on the top boundary:

$$
\bar{\phi}_{\mathrm{top}}
= \frac{1}{N}
  \sum_{k=1}^{n_z-1}
  \sum_{i=1}^{n_x-1} \phi_{i,n_y,k}
$$

where

$$
N = (n_x-1)(n_z-1)
$$

Therefore,

$$
Nu = \frac{1}
{\bar{\phi}_{\mathrm{bottom}}-\bar{\phi}_{\mathrm{top}}}
$$