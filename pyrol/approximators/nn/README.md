# Networks
## Deep Q Networks
Q-Learning by Chris Watkins
```text
@article{watkins1992q,
  title={Q-learning},
  author={Watkins, Christopher JCH and Dayan, Peter},
  journal={Machine learning},
  volume={8},
  number={3-4},
  pages={279--292},
  year={1992},
  publisher={Springer}
}
```

Chris Watkins Built off of Richard Bellman's Dynamic Programming and the Bellman Equations
```text
@article{bellman1966dynamic,
  title={Dynamic programming},
  author={Bellman, Richard},
  journal={Science},
  volume={153},
  number={3731},
  pages={34--37},
  year={1966},
  publisher={American Association for the Advancement of Science}
}
```

##### torch.nn Notes
-nn.Linear already has an initialization of
Linear.weight = U(- k**0.5, k **0.5), where k = 1/ in_features, where in_features are size
of each input sample
y = x A.' + b, notice that A is transposed  
-Linear.bias is the same
-Sequential is useful for connecting layers dyanmically
-ModuleList is useful for dynamically connecting layers but doesn't automatically pass into `forward` there fore you can
probe or use intermediate layer results like in an lstm
