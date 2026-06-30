from evopoint_da._compat import alias_children, alias_module

_module = alias_module(__name__, "protcross.cli")
alias_children(
    __name__,
    "protcross.cli",
    ("download_af2", "main", "map_labels", "predict", "preprocess", "setup_assets", "train"),
)
globals().update(_module.__dict__)
