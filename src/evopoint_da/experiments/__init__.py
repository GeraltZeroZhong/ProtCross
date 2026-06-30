from evopoint_da._compat import alias_children, alias_module

_module = alias_module(__name__, "protcross.experiments")
alias_children(__name__, "protcross.experiments", ("multiseed_benchmark", "strategy_search"))
globals().update(_module.__dict__)
