from toolz.functoolz import curry, compose
import pandas
from sklearn.base import MetaEstimatorMixin
import numpy as np
from toolz.curried import valfilter, valmap
from itertools import starmap
from operator import __mul__, methodcaller, __add__
from sklearn2code.sym.base import sym_predict
from sklearn2code.sym.function import cart
from sklearn2code.sym.expression import RealNumber
from six.moves import reduce

def notnone(x):
    return x is not None

@curry
def growd(d, x):
    shape = x.shape
    l = len(shape)
    if l >= d:
        return x
    else:
        slice_args = ([slice(None)] * l) + [None] * (d-l)
        return x.__getitem__(slice_args)

@curry
def shrinkd(d, x):
    if isinstance(x, pandas.DataFrame):
        if x.shape[1] == 1:
            return x.iloc[:,0]
        else:
            return x
    shape = x.shape
    l = len(shape)
    if l <= d:
        return x
    else:
        slice_args = [slice(None) for _ in range(d)]
        hit = False
        for i in range(d, l):
            if shape[i] == 1 and not hit:
                slice_args.append(0)
            else:
                hit = True
                slice_args.append(slice(None))
        return x.__getitem__(slice_args)
    
def fit_predict(estimator, X, y=None, sample_weight=None, exposure=None):
    fit_args = {'X': X}
    predict_args = {'X': X}
    if y is not None:
        fit_args['y'] = y
    if sample_weight is not None:
        fit_args['sample_weight'] = sample_weight
    if exposure is not None:
        fit_args['exposure'] = exposure
        predict_args['exposure'] = exposure
    if hasattr(estimator, 'fit_predict'):
        return estimator.fit_predict(**fit_args)
    else:
        estimator.fit(**fit_args)
        return estimator.predict(**predict_args)

class LinearCombination(MetaEstimatorMixin):
    def __init__(self, estimators, coefficients):
        self.estimators = estimators
        self.coefficients = coefficients
        if len(self.estimators) != len(self.coefficients):
            raise ValueError('Number of estimators does not match number of coefficients.')
    
    def fit(self, X, y=None, sample_weight=None, exposure=None):
        raise NotImplementedError('Linear combinations should only be created after fitting.')
    
    def predict(self, X, exposure=None):
        data = valmap(growd(2), valfilter(notnone, dict(X=X, exposure=exposure)))
        prediction = self.coefficients[0]*self.estimators[0].predict(**data)
        if len(prediction.shape) == 2 and prediction.shape[1] == 1:
            prediction = np.ravel(prediction)
            ravel = True
        elif len(prediction.shape) == 1:
            ravel = True
        else:
            ravel = False
        for i, estimator in enumerate(self.estimators[1:]):
            
            prediction += self.coefficients[i+1] * (np.ravel(estimator.predict(**data)) if ravel else estimator.predict(**data))
        return prediction
    
    def transform(self, X, exposure=None):
        data = valmap(growd(2), valfilter(notnone, dict(X=X, exposure=exposure)))
        return np.concatenate(tuple(map(compose(growd(2), methodcaller('predict', **data)), self.estimators)), axis=1)
        
    def sym_predict(self):
        return reduce(__add__, starmap(__mul__, zip(map(RealNumber, self.coefficients), map(sym_predict, self.estimators))))
    
    def sym_transform(self):
        return cart(*map(sym_predict, self.estimators))
    
    def __mul__(self, factor):
        estimators = [est for est in self.estimators]
        coefficients = [coeff * factor for coeff in self.coefficients]
        return LinearCombination(estimators, coefficients)
    
    def __add__(self, other):
        if isinstance(other, LinearCombination):
            estimators = self.estimators + other.estimators
            coefficients = self.coefficients + other.coefficients
            return LinearCombination(estimators, coefficients)
        else:
            return self + LinearCombination([other], [1.0])
    
