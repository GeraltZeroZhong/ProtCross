from ._compat import alias_module

globals().update(alias_module(__name__, "protcross.assets").__dict__)
