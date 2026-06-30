from evopoint_da._compat import alias_children, alias_module

_module = alias_module(__name__, "protcross.evaluation")
alias_children(__name__, "protcross.evaluation", ("adaptive", "metrics"))
globals().update(_module.__dict__)
