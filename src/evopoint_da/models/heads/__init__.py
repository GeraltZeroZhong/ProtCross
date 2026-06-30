from evopoint_da._compat import alias_children, alias_module

_module = alias_module(__name__, "protcross.models.heads")
alias_children(__name__, "protcross.models.heads", ("classifier",))
globals().update(_module.__dict__)
