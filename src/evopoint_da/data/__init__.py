from evopoint_da._compat import alias_children, alias_module

_module = alias_module(__name__, "protcross.data")
alias_children(
    __name__,
    "protcross.data",
    ("af2", "components", "datamodule", "dataset", "esm", "label_mapping", "pca", "preprocess", "structure"),
)
globals().update(_module.__dict__)
