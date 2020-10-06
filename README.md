# CGCNN

## New options

 __--original True__ - Use original convolution function (eq.5 from Grossman paper https://arxiv.org/pdf/1710.10324.pdf), default = False. If original = False cgcnn uses 2-layer NN with softplus activation instead of eq.5.

__--nn_pool True__ - Use Tony's pooling, default = False.

 __--orb True__ - Use orbitals as atomic features, default = False.
