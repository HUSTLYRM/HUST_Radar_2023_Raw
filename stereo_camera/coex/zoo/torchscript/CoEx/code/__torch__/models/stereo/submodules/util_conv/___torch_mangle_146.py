class BasicConv(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  relu : bool
  use_bn : bool
  conv : __torch__.torch.nn.modules.conv.___torch_mangle_145.ConvTranspose2d
  bn : __torch__.torch.nn.modules.batchnorm.BatchNorm2d
  LeakyReLU : __torch__.torch.nn.modules.activation.LeakyReLU
  def forward(self: __torch__.models.stereo.submodules.util_conv.___torch_mangle_146.BasicConv,
    x: Tensor) -> Tensor:
    x0 = (self.conv).forward(x, None, )
    if self.use_bn:
      x1 = (self.bn).forward(x0, )
    else:
      x1 = x0
    if self.relu:
      x2 = (self.LeakyReLU).forward(x1, )
    else:
      x2 = x1
    return x2
