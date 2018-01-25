from numpy.testing.utils import assert_array_almost_equal
from pandas.core.frame import DataFrame
from sklearn2code.sym.function import tupify
from sklearntoolsbase.sklearntoolsbase import growd

def assert_correct_exported_module(estimator, module, methods, estimator_data, module_data):
    for method in methods:
        estimator_output = getattr(estimator, method)(**estimator_data)
        module_output = getattr(module, method)(**module_data)
        if not isinstance(module_output, tuple):
            module_output = (module_output,)
        assert_array_almost_equal(growd(2, estimator_output), DataFrame({i: col for i, col in enumerate(tupify(module_output))}))