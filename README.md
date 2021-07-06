# Variance Reduced Optimizers in PyTorch

This repo contains implementation of SVRG, SARAH (SpiderBoost), SCSG and  Geom-SARAH algorithms based on PyTorch. It was used to produce experiments for the paper [**Adaptivity of Stochastic Gradient Methods for Nonconvex Optimization**](https://arxiv.org/pdf/2002.05359.pdf) (Horvath et al., 2020).  

To replicate our experiments, first, recreate `conda` environment from [environment.yml](environment.yml). Run scripts are available in [runs/](runs) directory.


If you find this useful, please consider citing: 

```
@article{horvath2020adaptivity,
  title={Adaptivity of stochastic gradient methods for nonconvex optimization},
  author={Horv{\'a}th, Samuel and Lei, Lihua and Richt{\'a}rik, Peter and Jordan, Michael I},
  journal={arXiv preprint arXiv:2002.05359},
  year={2020}
}
```

## References: 
- **SVRG**: [Accelerating Stochastic Gradient Descent using
Predictive Variance Reduction](https://papers.nips.cc/paper/2013/file/ac1dd209cbcc5e5d1c6e28598e8cbbe8-Paper.pdf) (Johnson and Zhang, 2013)
- **SCSG**: [Less than a single pass: Stochastically controlled stochastic gradient method.](https://arxiv.org/abs/1706.09156) (Lei and Jordan, 2017)
- **SARAH**: [SARAH: A novel method for machine learning problems using stochastic recursive gradient](http://proceedings.mlr.press/v70/nguyen17b/nguyen17b.pdf) (Nguyen et al., 2017)
- **SpiderBoost**: [SpiderBoost and Momentum: Faster Stochastic Variance Reduction Algorithms](https://arxiv.org/abs/1810.10690) (Wang et al., 2019)
- **Geom-SARAH**: [Adaptivity of Stochastic Gradient Methods for Nonconvex Optimization](https://arxiv.org/pdf/2002.05359.pdf) (Horvath et al., 2020)
