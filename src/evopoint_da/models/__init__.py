from evopoint_da._compat import alias_children, alias_module

_module = alias_module(__name__, "protcross.models")
alias_children(
    __name__,
    "protcross.models",
    ("domain_weights", "module", "backbones", "backbones.pointnet2", "heads", "heads.classifier"),
)
globals().update(_module.__dict__)
