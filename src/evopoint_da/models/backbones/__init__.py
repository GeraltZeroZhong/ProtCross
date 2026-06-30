from evopoint_da._compat import alias_children, alias_module

_module = alias_module(__name__, "protcross.models.backbones")
alias_children(__name__, "protcross.models.backbones", ("pointnet2",))
globals().update(_module.__dict__)
