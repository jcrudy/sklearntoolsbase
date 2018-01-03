'''
Created on Feb 11, 2016

@author: jason
'''
from sklearn.base import BaseEstimator, clone, MetaEstimatorMixin
import numpy as np
from six import with_metaclass
from functools import update_wrapper
import sys
from pandas.core.indexing import IndexingError
if sys.version_info[0] < 3:
    from inspect import getargspec
else:
    from inspect import getfullargspec
    from collections import namedtuple
    ArgSpec = namedtuple('ArgSpec', ['args', 'varargs', 'keywords', 'defaults'])
    def getargspec(*args, **kwargs):
        return ArgSpec(*getfullargspec(*args, **kwargs)[:4])
from . import __version__
from toolz.functoolz import curry
from sklearn.exceptions import NotFittedError
from six.moves import reduce
import pandas
# 
# def if_delegate_has_method(*args, **kwargs):
#     return decorator(sklearn_if_delegate_has_method(*args, **kwargs))

# @decorator
def if_delegate_has_method(*args, **kwargs):
    return lambda x: x
#     return fn(*args, **kwargs)

def safe_call(fn, args):
    if hasattr(fn, '_spec'):
        spec = fn._spec
    else:   
        spec = getargspec(fn)
    if spec.keywords is not None:
        return fn(**args)
    else:
        safe_args = {arg: args[arg] for arg in spec.args[1:] if arg in args}
        return fn(**safe_args)

def safer_call(fn, *args, **kwargs):
    kwargs = kwargs.copy()
    if hasattr(fn, '_spec'):
        spec = fn._spec
    else:
        try:
            spec = getargspec(fn)
        except TypeError:
            spec = getargspec(fn.__call__)
    spec_args = list(spec.args)
    if spec_args and spec_args[0] == 'self':
        spec_args = spec_args[1:]
    if spec.varargs is None:
        for i, arg in enumerate(args):
            name = spec_args[i]
            kwargs[name] = arg
            args = []
    if spec.keywords is None:
        for name in list(kwargs.keys()):
            if name not in spec_args:
                del kwargs[name]
    return fn(*args, **kwargs)


def make2d(x):
    if len(x.shape) == 1:
        return x[:, None]
    return x

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
            return x.ix[:,0]
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

def safe_rows_select(data, rows):
    if hasattr(data, 'loc'):
        if len(data.shape) > 1:
            return data.loc[rows,:]
        else:
            return data.loc[rows]    
    elif len(data.shape) > 1:
        return data[rows, :]
    else:
        return data[rows]

def safe_column_select(data, col):
    if hasattr(data, 'loc'):
        return data.loc[:, col]
    else:
        return data[:, col]

def _subset(data, idx):
    if len(data.shape) == 1:
        return data[idx]
    else:
        if hasattr(data, 'iloc'):
            return data.iloc[idx, :].reset_index(drop=True)
        else:
            return data[idx, :]

def _subset_data(data, idx):
    result = {}
    for k in data.keys():
        result[k] = _subset(data[k], idx)
    return result

def safe_assign_subset(arr, idx, value):
    try:
        arr.loc[idx, :] = value
    except AttributeError:
        try:
            arr[idx, :] = value.reshape(arr[idx, :].shape)
        except:
            arr.flat[idx] = value
    except IndexingError:
        arr[idx] = value
        
def safe_assign_column(arr, col, value):
    if hasattr(arr, 'loc'):
        arr.loc[:, col] = value
    else:
        arr[:, col] = value


def safe_column_names(arr):
    if hasattr(arr, 'columns'):
        return list(arr.columns)
    elif len(arr.shape) == 2:
        return list(map(lambda i: 'x%d'%i, range(arr.shape[1])))
    elif len(arr.shape) == 1:
        return ['x']
    else:
        raise ValueError()

@curry
def clean_column_name(xlabels, col):
    if isinstance(col, int):
        return xlabels[col]
    else:
        return col
    
def _fit_and_score(estimator, data, scorer, train, test):
    train_data = _subset_data(data, train)
    estimator_ = clone(estimator).fit(**train_data)
    test_data = _subset_data(data, test)
    score = safer_call(scorer, estimator_, **test_data)
    return (score, np.sum(test), estimator_)

def _fit_and_predict(estimator, data, train, test):
    train_data = _subset_data(data, train)
    estimator_ = clone(estimator).fit(**train_data)
    test_data = _subset_data(data, test)
    prediction = safer_call(estimator_.predict, **test_data)
    return estimator_, prediction, test

class SklearnTool(object):
    _version = __version__
# 
# def name_estimator(estimator):
#     if hasattr(estimator, 'name'):
#         return estimator.name
#     else:
#         return estimator.__class__.__name__
# 
# def combine_named_estimators(names):
#     used_set = set()
#     result = []
#     for name in names:
#         new_name = name
#         i = 2
#         while new_name in used_set:
#             new_name = name + '_' + str(i)
#             i += 1
#             if i > 1e5:
#                 raise ValueError('Unable to name estimator %s in pipeline' % str(name))
#         used_set.add(new_name)
#         result.append(new_name)
#     return result

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

class STEstimator(BaseEstimator, SklearnTool):
    def __sub__(self, other):
        '''
        self - other
        '''
        return self + (-1. * other)
    
    def __rsub__(self, other):
        '''
        other - self
        '''
        return other + (-1. * self)
    
    def __rmul__(self, factor):
        return self.__mul__(factor)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __mul__(self, factor):
        return LinearCombination([self], [factor])
    
    def __add__(self, other):
        return (1.0 * self) + other
    
    def _process_args(self, **kwargs):
        result = {}
        for k, v in kwargs.items():
            if v is not None:
                result[k] = v
        for k in result.keys():
            v = result[k]
            if isinstance(v, np.ndarray):
                if len(v.shape) == 1:
                    result[k] = v[:, None]
        return result
    
#     def mask(self, mask):
#         return mask_estimator(self, mask)

    def __and__(self, other):
        '''
        self & other
        '''
        return MultiEstimator([self]) & other
    
    def __rand__(self, other):
        '''
        other & self
        '''
        return MultiEstimator([other]) & self


class StagedEstimator(STEstimator, MetaEstimatorMixin):
    def __init__(self, stages):
        self.stages = stages
        self.intermediate_stages = self.stages[:-1]
        self.final_stage = self.stages[-1]
    
    def __rshift__(self, other):
        new_stages = [stage for stage in self.stages]
        if isinstance(other, StagedEstimator):
            new_stages += other.stages
        else:
            new_stages += [other]
        return StagedEstimator(new_stages)
            
    def __rrshift__(self, other):
        new_stages = self.stages.copy()
        if isinstance(other, StagedEstimator):
            new_stages = other.stages + new_stages
        else:
            new_stages = [other] + new_stages
        return StagedEstimator(new_stages)
    
    def _transform_args(self, data):
        result = {'X': data['X']}
        if 'exposure' in data:
            result['exposure'] = data['exposure']
        return result
    
    def _update(self, data):
        for stage in self.intermediate_stages_:
            try:
                # Stage knows to discard whatever it doesn't need
                stage.update(data)
            except AttributeError:
                data['X'] = safe_call(stage.transform, self._transform_args(data))
    
    def fit_update(self, data):
        self.intermediate_stages_ = []
        for stage in self.intermediate_stages:
            # Stage knows to discard whatever it doesn't need
            stage_ = clone(stage)
            safe_call(stage_.fit, data)
            try:
                stage_.update(data)
            except AttributeError:
                try:
                    data['X'] = safe_call(stage_.transform, self._transform_args(data))
                except:
                    data['X'] = safe_call(stage_.transform, self._transform_args(data))
            self.intermediate_stages_.append(stage_)
        return data
    
    def fit(self, X, y=None, sample_weight=None, exposure=None):
        data = self.fit_update(self._process_args(X=X, y=y, sample_weight=sample_weight, exposure=exposure))
        self.final_stage_ = safe_call(clone(self.final_stage).fit, data)
        return self
    
    def fit_predict(self, X, y=None, sample_weight=None, exposure=None):
        data = self.fit_update(self._process_args(X=X, y=y, sample_weight=sample_weight, exposure=exposure))
        self.final_stage_ = clone(self.final_stage)
        return safe_call(self.final_stage_.fit_predict, data)
    
    def transform(self, X, exposure=None):
        data = self._process_args(X=X, exposure=exposure)
        self._update(data)
        return safe_call(self.final_stage_.transform, data)
    
    def predict(self, X, exposure=None):
        data = self._process_args(X=X, exposure=exposure)
        self._update(data)
        try:
            return safe_call(self.final_stage_.predict, data)
        except:
            return safe_call(self.final_stage_.predict, data)
        
    def predict_proba(self, X, exposure=None):
        data = self._process_args(X=X, exposure=exposure)
        self._update(data)
        return safe_call(self.final_stage_.predict_proba, data)
    
    def predict_log_proba(self, X, exposure=None):
        data = self._process_args(X=X, exposure=exposure)
        self._update(data)
        return safe_call(self.final_stage_.predict_log_proba, data)
    
    def score(self, X, y=None, sample_weight=None, exposure=None):
        data = self._process_args(X=X, exposure=exposure)
        self._update(data)
        return safe_call(self.final_stage_.score, data)
    
    def decision_function(self, X, exposure=None):
        data = self._process_args(X=X, exposure=exposure)
        self._update(data)
        return safe_call(self.final_stage_.decision_function, data)

def staged(estimator):
    return StagedEstimator([estimator])
        
# def as_pipeline(estimator):
#     try:
#         return estimator.as_pipeline()
#     except AttributeError:
#         return STPipeline([(name_estimator(estimator), estimator)])

class STSimpleEstimator(STEstimator):
    def __rshift__(self, other):
        '''
        self >> other
        '''
        return staged(self) >> other
        
    def __rrshift__(self, other):
        '''
        other >> self
        '''
        return staged(other) >> self
        
    def __lshift__(self, other):
        '''
        '''
        return staged(other) >> self
        
    def __rlshift__(self, other):
        '''
        other << self
        '''
        return staged(self) >> other

#     def as_pipeline(self):
#         return STPipeline([(name_estimator(self), self)])
        
# class STPipeline(STEstimator, Pipeline):
#     def as_pipeline(self):
#         return self
#     
#     def __rshift__(self, other):
#         '''
#         self >> other
#         '''
#         other = as_pipeline(other)
#         steps = self.steps + other.steps
#         names = [step[0] for step in steps]
#         estimators = [step[1] for step in steps]
#         return STPipeline(zip(combine_named_estimators(names), estimators))
class LinearCombination(STSimpleEstimator, MetaEstimatorMixin):
    def __init__(self, estimators, coefficients):
        self.estimators = estimators
        self.coefficients = coefficients
        assert len(self.estimators) == len(self.coefficients)
    
    def fit(self, X, y=None, sample_weight=None, exposure=None):
        raise NotImplementedError('Linear combinations should only be created after fitting.')
    
    def predict(self, X, exposure=None):
        data = self._process_args(X=X, exposure=exposure)
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

class _BasicDelegateDescriptor(object):
    def __init__(self, fn, delegate_name):
        self.fn = fn
        self.delegate_name = delegate_name
        self.method_name = fn.__name__
        # update the docstring of the descriptor
        update_wrapper(self, fn)
        
    def __get__(self, obj, type=None):  # @ReservedAssignment
        if self.delegate_name is None:
            try:
                delegate_name = obj._delegates[self.method_name]
            except KeyError:
                try:
                    delegate_name = obj.__class__._class_delegates[self.method_name]
                except KeyError:
                    raise AttributeError()
            except AttributeError:
                raise NotFittedError()
        else:
            delegate_name = self.delegate_name
        clone_name = delegate_name + '_'
        
        # If the clone doesn't exist, it needs to be created by this call
        if hasattr(obj, clone_name):
            delegate = getattr(obj, clone_name)
            method = getattr(delegate, self.method_name)
            spec = getargspec(method)
            def out(*args, **kwargs):
                return method(*args, **kwargs)
            out._spec = spec
        elif hasattr(obj, delegate_name):
            delegate = getattr(obj, delegate_name)
            method = getattr(delegate, self.method_name)
            spec = getargspec(method)
            def out(*args, **kwargs):
                setattr(obj, clone_name, method(*args, **kwargs))
                return obj
            out._spec = spec
        else:
            raise NotFittedError()
        update_wrapper(out, self.fn)
        return out

def delegate_by_name(delegate_name=None):
    return lambda fn: _BasicDelegateDescriptor(fn, delegate_name)

def delegate(fn):
    return _BasicDelegateDescriptor(fn, None)

# def delegate_init(fn):
#     def init(self, *args, **kwargs):
#         self._delegates = {}
#         fn(self, *args, **kwargs)
#     update_wrapper(init, fn)
#     return init

class DelegatingMetaClass(type):
    '''
    Every subclass gets its own _delegates dictionary.  If the dictionary were
    just a normal class attribute on BaseDelegatingEstimator, it would be shared
    among all subclasses.
    '''
    def __init__(cls, name, bases, dict):  # @ReservedAssignment
        super(DelegatingMetaClass, cls).__init__(name, bases, dict)
        cls._class_delegates = {}
#         cls.__init__ = delegate_init(cls.__init__)
        
#     def __call__(cls, *args, **kwargs):  # @NoSelf
#         obj = super(DelegatingMetaClass, cls).__call__(*args, **kwargs)
#         obj._delegates = {}

standard_methods = ['fit', 'predict', 'score', 'predict_proba', 'decision_function', 
                         'predict_log_proba', 'transform', 'fit_predict']
non_fit_methods = ['predict', 'score', 'predict_proba', 'decision_function', 
                         'predict_log_proba', 'transform']
predict_methods = ['predict', 'predict_proba', 'decision_function', 
                         'predict_log_proba']
sym_methods = ['syms', 'sym_predict', 'sym_transform', 
               'sym_predict_parts', 'sym_transform_parts', 'sym_predict_proba',
               'sym_predict_proba_parts']


class BaseDelegatingEstimator(with_metaclass(DelegatingMetaClass, STSimpleEstimator, MetaEstimatorMixin)):
    def _create_delegates(self, name, method_names):
        if not hasattr(self, '_delegates'):
            self._delegates = {}
#         delegate_ = getattr(self, name)
#         methods = [method for method in method_names if callable(getattr(delegate_, method, None))]
        for method in method_names:
            self._delegates[method] = name
#             def fn(obj):
#                 pass
#             fn.__name__ = method
#             setattr(self, method, delegate()(MethodType(fn, self, self.__class__)))

    @delegate
    def fit(self):
        pass
     
    @delegate
    def predict(self):
        pass
     
    @delegate
    def score(self):
        pass
     
    @delegate
    def predict_proba(self):
        pass
     
    @delegate
    def decision_function(self):
        pass
     
    @delegate
    def predict_log_proba(self):
        pass
     
    @delegate
    def transform(self):
        pass
    
    @delegate
    def syms(self):
        pass
    
    @delegate
    def sym_predict(self):
        pass
    
    @delegate
    def sym_transform(self):
        pass
    
    @delegate
    def sym_predict_proba(self):
        pass
    
    @delegate
    def sym_predict_parts(self):
        pass
    
    @delegate
    def sym_transform_parts(self):
        pass
    
    @delegate
    def sym_predict_proba_parts(self):
        pass
    

class DelegatingEstimator(BaseDelegatingEstimator):
    _delegates = {'fit': 'estimator', 'predict': 'estimator', 'score': 'estimator', 
                  'predict_proba': 'estimator', 'decision_function': 'estimator', 
                  'predict_log_proba': 'estimator', 'transform': 'estimator'}
    def __init__(self, estimator):
        self.estimator = estimator

def st(estimator):
    return DelegatingEstimator(estimator)

# class EstimatorStage(DelegatingEstimator):
#     def __init__(self, estimator, method_args):
#         self.estimator = estimator
#         self._create_delegates('estimator', standard_methods)
#         
#     def update(self, data):
#         '''
#         Update data in place to pass to the next stage in a pipeline.
#         '''
# class TransformerStage(EstimatorStage):
#     def update(self, data):
#         data['X'] = self.transform(**data)
        
# class ConcatenatingResponseStage(EstimatorStage):
#     def update(self, data):
#         data[''] = self.transform(**data)
class Wrapper(object):
    def __init__(self, content):
        self.content = content

class AlreadyFittedEstimator(DelegatingEstimator):
    def __init__(self, estimator):
        if isinstance(estimator, Wrapper):
            self.estimator = estimator
        else:
            self.estimator = Wrapper(estimator)
#         self.estimator_ = self.estimator.content
        self._create_delegates('estimator', non_fit_methods)
    
    @property
    def estimator_(self):
        return self.estimator.content
    
    def fit(self, X, y=None, sample_weight=None, exposure=None):
        return self
    
class BoundedEstimator(DelegatingEstimator):
    def __init__(self, estimator, lower_bound=float('-inf'), upper_bound=float('inf')):
        self.estimator = estimator
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        # Delegate everything except predict and score
        self._create_delegates('estimator', ['fit', 'predict_proba', 'decision_function', 
                         'predict_log_proba', 'transform'])
    
    def predict(self, X, exposure=None):
        data = self._process_args(X=X, exposure=exposure)
        raw_prediction = self.estimator_.predict(**data)
        bounded_prediction = np.maximum(np.minimum(raw_prediction, self.upper_bound), self.lower_bound)
        return bounded_prediction
        
    @if_delegate_has_method('estimator')
    def predict_proba(self, X, exposure=None):
        data = self._process_args(X=X, exposure=exposure)
        return self.estimator_.predict_proba(**data)
    
    @if_delegate_has_method('estimator')
    def predict_log_proba(self, X, exposure=None):
        data = self._process_args(X=X, exposure=exposure)
        return self.estimator_.predict_log_proba(**data)
    
    @if_delegate_has_method('estimator')
    def decision_function(self, X, exposure=None):
        data = self._process_args(X=X, exposure=exposure)
        return self.estimator_.decision_function(**data)
#     def predict(self):
#     
# def mask_estimator(estimator, mask):
#     return MaskedEstimator(estimator, mask)
#     try:
#         return estimator.mask(mask)
#     except AttributeError:
#         return MultipleResponseEstimator([(name_estimator(estimator), mask, estimator)])

# def is_masked(estimator):
#     return isinstance(estimator, MultipleResponseEstimator)

def compatible_masks(masks):
    # (min, max)
    intersection = (0, float('inf'))
    for mask in masks:
        current_mask = mask
        if isinstance(current_mask, np.ndarray):
            if current_mask.dtype.kind == 'b':
                current = (current_mask.shape[0], current_mask.shape[0])
            else:
                current = (np.max(current_mask), float('inf'))
        elif isinstance(current_mask, slice):
            current = (max(current_mask.start, current_mask.stop), float('inf'))
        intersection = (max(current[0], intersection[0]), min(current[1], intersection[1]))
        
    if intersection[0] > intersection[1]:
        return False
    else:
        return True
    
def convert_mask(mask):
    if not (isinstance(mask, np.ndarray) or isinstance(mask, slice)) \
            and hasattr(mask, '__iter__'):
        return np.array(mask)
    else:
        return mask

def safe_col_select(data, cols):
    if hasattr(data, 'loc'):
        return data.loc[:, cols]
    else:
        return data[:, cols]
        

class BaseRowSubsetTransformer(STSimpleEstimator):
    '''
    Removes some rows for whatever reason.
    '''
    def fit(self, X=None, y=None, sample_weight=None, exposure=None):
        return self
    
    def transform(self, X=None, y=None, sample_weight=None, exposure=None):
        data = self._process_args(X=X, y=y, sample_weight=sample_weight, 
                                  exposure=exposure)
        rows = self._predicate(data)
        return _subset(X, rows)
    
    def update(self, data):
        rows = self._predicate(data)
        for k in data.keys():
            data[k] = _subset(data[k], rows)

class BaseRowSubsetFitter(DelegatingEstimator):
    def __init__(self, estimator):
        self.estimator = estimator
        self._create_delegates('estimator', standard_methods)
    
    def fit(self, X, y=None, sample_weight=None, exposure=None):
        data = self._process_args(X=X, y=y, sample_weight=sample_weight, 
                                  exposure=exposure)
        self.estimator_ = clone(self.estimator).fit(**(_subset_data(data, self._predicate(data))))
        return self

class ArgumentFixingEstimator(STSimpleEstimator, MetaEstimatorMixin):
    def __init__(self, estimator, arg_dict):
        self.estimator = estimator
        self.arg_dict = arg_dict
#         self._create_delegates('estimator', non_fit_methods)
    
    @if_delegate_has_method('estimator')
    def fit(self, X, y=None, sample_weight=None, exposure=None):
        data = self._process_args(X=X, y=y, sample_weight=sample_weight,
                                  exposure=exposure)
        if 'fit' in self.arg_dict:
            data.update(self.arg_dict['fit'])
        self.estimator_ = clone(self.estimator).fit(**data)
        return self
    
    @if_delegate_has_method('estimator')
    def predict(self, X, exposure=None):
        data = self._process_args(X=X, exposure=exposure)
        if 'predict' in self.arg_dict:
            data.update(self.arg_dict['predict'])
        return self.estimator_.predict(**data)
    
    @if_delegate_has_method('estimator')
    def predict_proba(self, X, exposure=None):
        data = self._process_args(X=X, exposure=exposure)
        if 'predict_proba' in self.arg_dict:
            data.update(self.arg_dict['predict_proba'])
        return self.estimator_.predict_proba(**data)
    
    @if_delegate_has_method('estimator')
    def predict_log_proba(self, X, exposure=None):
        data = self._process_args(X=X, exposure=exposure)
        if 'predict_log_proba' in self.arg_dict:
            data.update(self.arg_dict['predict_log_proba'])
        return self.estimator_.predict_log_proba(**data)
    
    @if_delegate_has_method('estimator')
    def decision_function(self, X, exposure=None):
        data = self._process_args(X=X, exposure=exposure)
        if 'decision_function' in self.arg_dict:
            data.update(self.arg_dict['decision_function'])
        return self.estimator_.decision_function(**data)

def non_null_rows(arr):
    if hasattr(arr, 'notnull'):
        return arr.notnull().any(axis=1)
    else:
        return ~(np.isnan(arr).any(axis=1))

def non_null_rows_dict(data):
    result = None
    for v in data.values():
        if result is None:
            result = np.ones(shape=v.shape[0], dtype=bool)
        result &= non_null_rows(v)
    if result is None:
        result = slice(None)
    return result

class NonMissingRowSubsetMixin(object):
    def _predicate(self, data):
        return non_null_rows_dict(data)
    
class NonNullSubsetFitter(BaseRowSubsetFitter, NonMissingRowSubsetMixin):
    pass
    
class ColumnSubsetTransformer(STSimpleEstimator):
    '''
    Takes all data from X and splits it into X, y, sample_weight, and exposure.  Use with 
    StagedEstimator.  If used as transformer, only gives X (with appropriate subset of columns).
    '''
    def __init__(self, x_cols=slice(None), y_cols=None,
                  sample_weight_cols=None, exposure_cols=None):
        self.x_cols = x_cols
        self.y_cols = y_cols
        self.sample_weight_cols = sample_weight_cols
        self.exposure_cols = exposure_cols
        
    def fit(self, X=None, y=None, sample_weight=None, exposure=None):
        return self
    
    def transform(self, X=None, y=None, sample_weight=None, exposure=None):
        return safe_col_select(X, self.x_cols)
    
    def update(self, args):
        keys = {'X':self.x_cols, 'y':self.y_cols, 'sample_weight':self.sample_weight_cols, 
                'exposure':self.exposure_cols}
        X = args['X']
        for key, cols in keys.items():
            if cols is not None:
                try:
                    args[key] = safe_col_select(X, cols)
                except KeyError:
                    if key in {'X', 'exposure'}:
                        raise
                    else:
                        pass
    
class MaskedEstimator(STSimpleEstimator, MetaEstimatorMixin):
    def __init__(self, estimator, mask):
        self.estimator = estimator
        self.mask = convert_mask(mask)
    
    def _mask_y(self, y):
        if y is None:
            return None
        result = y[:, self.mask]
        if len(result) == 0:
            return None
        return result
    
    def fit(self, X, y=None, sample_weight=None, exposure=None):
        data = self._process_args(X=X, y=self._mask_y(y), sample_weight=sample_weight,
                                  exposure=exposure)
        if len(data['y'].shape) > 1 and data['y'].shape[1] == 1:
            data['y'] = np.ravel(data['y'])
        self.estimator_ = clone(self.estimator).fit(**data)
        return self
        
    @if_delegate_has_method('estimator')
    def score(self, X, y=None, sample_weight=None, exposure=None):
        data = self._process_args(X=X, y=self._mask_y(y), sample_weight=sample_weight,
                                  exposure=exposure)
        return self.estimator_.score(**data)
    
    @if_delegate_has_method('estimator')
    def predict(self, X, exposure=None):
        data = self._process_args(X=X, exposure=exposure)
        return self.estimator_.predict(**data)
    
    @if_delegate_has_method('estimator')
    def predict_proba(self, X, exposure=None):
        data = self._process_args(X=X, exposure=exposure)
        return self.estimator_.predict_proba(**data)
    
    @if_delegate_has_method('estimator')
    def predict_log_proba(self, X, exposure=None):
        data = self._process_args(X=X, exposure=exposure)
        return self.estimator_.predict_log_proba(**data)
    
    @if_delegate_has_method('estimator')
    def decision_function(self, X, exposure=None):
        data = self._process_args(X=X, exposure=exposure)
        return self.estimator_.decision_function(**data)
        
class MultiEstimator(STSimpleEstimator, MetaEstimatorMixin):
    def __init__(self, estimators):
        self.estimators = estimators
    
    def __and__(self, other):
        new_estimators = [est for est in self.estimators]
        if isinstance(other, MultiEstimator):
            new_estimators += other.estimators
        else:
            new_estimators += [other]
        return MultiEstimator(new_estimators)
            
    def fit(self, X, y=None, sample_weight=None, exposure=None):
        args = self._process_args(X=X, y=y, sample_weight=sample_weight,
                                  exposure=exposure)
        self.estimators_ = []
        for estimator in self.estimators:
            self.estimators_.append(clone(estimator).fit(**args))
        return self
    
    def transform(self, X, y=None, sample_weight=None, exposure=None):
        args = self._process_args(X=X, exposure=exposure)
        results = []
        total_cols = 0
        for estimator in self.estimators_:
            result = estimator.transform(**args)
            try:
                total_cols += result.shape[1]
            except IndexError:
                total_cols += 1
            if len(result.shape) == 1:
                result = result[:, None]
            results.append(result)
            
        result = np.concatenate(results, axis=1)
        assert result.shape[1] == total_cols
        return result
    
    def predict(self, X, exposure=None):
        args = self._process_args(X=X, exposure=exposure)
        results = []
        for estimator in self.estimators_:
            result = estimator.predict(**args)
            if len(result.shape) == 1:
                result = result[:, None]
            results.append(result)
        return np.concatenate(results, axis=1)
    
    def predict_proba(self, X, exposure=None):
        args = self._process_args(X=X, exposure=exposure)
        results = []
        for estimator in self.estimators_:
            result = estimator.predict_proba(**args)
            if len(result.shape) == 1:
                result = result[:, None]
            results.append(result)
        return np.concatenate(results, axis=1)
    
    def predict_log_proba(self, X, exposure=None):
        args = self._process_args(X=X, exposure=exposure)
        results = []
        for estimator in self.estimators_:
            result = estimator.predict_log_proba(**args)
            if len(result.shape) == 1:
                result = result[:, None]
            results.append(result)
        return np.concatenate(results, axis=1)
    
    def decision_function(self, X, exposure=None):
        args = self._process_args(X=X, exposure=exposure)
        results = []
        for estimator in self.estimators_:
            result = estimator.decision_function(**args)
            if len(result.shape) == 1:
                result = result[:, None]
            results.append(result)
        return np.concatenate(results, axis=1)
    
    def score(self, X, y=None, sample_weight=None, exposure=None):
        args = self._process_args(X=X, y=y, sample_weight=sample_weight, exposure=exposure)
        results = []
        for estimator in self.estimators_:
            result = estimator.predict_log_proba(**args)
            results.append(result)
        return np.array(results, axis=1)

class DecisionPathTransformer(DelegatingEstimator):
    '''
    Just overrides transform to use decision_path instead.  Useful for pipelines.
    '''
    def __init__(self, estimator):
        self.estimator = estimator
        self._create_delegates('estimator', standard_methods)
    
    def transform(self, X, exposure=None):
        args = {'X': X}
        if exposure is not None:
            args['exposure'] = exposure
        result = self.estimator.decision_path(**args).todense()
        if len(result.shape) == 1:
            result = result[:, None]
        return result
