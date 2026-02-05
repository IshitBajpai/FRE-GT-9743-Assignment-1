import copy
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional

class InterpMethod(Enum):

    PIECEWISE_CONSTANT_LEFT_CONTINUOUS = 'PIECEWISE_CONSTANT_LEFT_CONTINUOUS'
    LINEAR = 'LINEAR'

    @classmethod
    def from_string(cls, value: str) -> 'InterpMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value

class ExtrapMethod(Enum):
    
    FLAT = 'FLAT'
    LINEAR = 'LINEAR'

    @classmethod
    def from_string(cls, value: str) -> 'ExtrapMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value

class Interpolator1D(ABC):

    def __init__(self,
                 axis1 : np.ndarray, 
                 values : np.ndarray, 
                 interpolation_method : InterpMethod,
                 extrpolation_method : ExtrapMethod) -> None:

        self.axis1_ = axis1
        self.values_ = values
        self.interp_method_ = interpolation_method
        self.extrap_method_ = extrpolation_method
        self.length_ = len(self.axis1)

    @abstractmethod
    def interpolate(self, x : float) -> float:
        pass

    @abstractmethod
    def integrate(self, start_x : float, end_x : float):
        pass

    @abstractmethod
    def gradient_wrt_ordinate(self, x : float):
        pass

    @abstractmethod
    def gradient_of_integrated_value_wrt_ordinate(self, start_x : float, end_x : float):
        pass
    
    @property
    def axis1(self) -> np.ndarray:
        return self.axis1_
    
    @property
    def values(self) -> np.ndarray:
        return self.values_
    
    @property
    def length(self) -> int:
        return self.length_

    @property
    def interp_method(self) -> str:
        return self.interp_method_.to_string()
    
    @property
    def extrap_method(self) -> str:
        return self.extrap_method_.to_string()

class Interpolator1DPCP(Interpolator1D):

    def __init__(self, axis1: np.ndarray, values: np.ndarray, extrpolation_method: ExtrapMethod) -> None:
        super().__init__(axis1, values, InterpMethod.LINEAR, extrpolation_method)
        assert self.extrap_method_ == ExtrapMethod.FLAT

    def interpolate(self, x: float) -> float:
        ### TODO
        if x<self.axis1_[0]:
            return self.values_[0]
        if x>self.axis1_[-1]:
            return self.values_[-1]
        
        idx = np.searchsorted(self.axis1_, x, side='right')

        return self.values_[idx]
    
    def gradient_wrt_ordinate(self, x : float):
        ### TODO
        gradient = np.zeros(self.length, dtype=float)
        if x>= self.axis1_[-1]:
            gradient[-1] = 1
            return gradient
        
        if x< self.axis1_[0]:
            gradient[0] = 1
            return gradient
        
        idx = np.searchsorted(self.axis1_, x, side='right')
        gradient[idx] = 1
        return gradient

    def integrate(self, start_x : float, end_x : float):
        ### TODO
        if start_x > end_x:
            start_x, end_x = end_x, start_x

        if start_x == end_x:
            return 0
        
        if self.length == 1:
            return (end_x - start_x) * self.values_[0]
        
        left_ = self.interpolate(start_x)
        right_ = self.interpolate(end_x)
        
        start_idx = np.searchsorted(self.axis1_, start_x, side='left')
        if start_idx == self.length:
            return (end_x - start_x)*left_
        start_area = (self.axis1_[start_idx] - start_x)*left_
        
        end_idx = np.searchsorted(self.axis1_, end_x, side='right') - 1
        if end_idx == -1:
            return (end_x - start_x) * right_
        
        end_area = (end_x - self.axis1_[end_idx])  *right_
        
        area = start_area + end_area

        for i in range (start_idx +1, end_idx+1):
            area += (self.axis1_[i]  - self.axis1_[i-1])*self.values_[i]
        
        area = area

        return area



    def gradient_of_integrated_value_wrt_ordinate(self, start_x : float, end_x : float):
        ### TODO

        if start_x > end_x:
            start_x, end_x = end_x, start_x
            
        gradient = np.zeros(self.length_)

        start_idx = np.searchsorted(self.axis1_, start_x, side='left')
        end_idx = np.searchsorted(self.axis1_, end_x, side='right') -1

        if start_idx == self.length_:
            gradient[-1] = end_x - start_x
            return gradient
        
        if end_idx == -1:
            gradient[0] = end_x - start_x
            return gradient
        
        gradient[start_idx] = self.axis1_[start_idx] - start_x
        
        for i in range(start_idx+1, end_idx+1):
            gradient[i] += self.axis1_[i] - self.axis1_[i-1]

        gradient[min(end_idx+1, self.length_-1)] += end_x - self.axis1_[end_idx]    
        return gradient

class InterpolatorFactory:

    @staticmethod
    def create_1d_interpolator(axis1 : np.ndarray | List, 
                               values : np.ndarray | List, 
                               interpolation_method : InterpMethod,
                               extrpolation_method : ExtrapMethod):


        axis1_ = copy.deepcopy(axis1)
        values_ = copy.deepcopy(values)
        if isinstance(axis1_, list):
            axis1_ = np.array(axis1_)
        if isinstance(values_, list):
            values_ = np.array(values_)
        assert len(axis1_.shape) == 1 and len(values_.shape) == 1
        assert len(axis1_) == len(values_)
        assert np.all(np.diff(axis1_) >= 0)
    
        if interpolation_method == InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS:
            return Interpolator1DPCP(axis1_, values_, extrpolation_method)
        else:
            raise Exception('Currently only support PCP interpolation')
