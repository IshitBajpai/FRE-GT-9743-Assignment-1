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
        axis = self.axis1_
        values = self.values_

        if x < axis[0]:
            return values[0]
        
        elif x > axis[-1]:
            return values[-1]
        
        insertion_index = np.searchsorted(axis, x, side="right")

        return values[insertion_index]
    
    def gradient_wrt_ordinate(self, x : float):
        gradient_vector = np.zeros(self.length, dtype=float)
        axis = self.axis1_

        if x >= axis[-1]:
            gradient_vector[-1] = 1.0
            return gradient_vector

        if x < axis[0]:
            gradient_vector[0] = 1.0
            return gradient_vector

        #  x lies within interpolation range
        insertion_index = np.searchsorted(axis, x, side="right")
        gradient_vector[insertion_index] = 1.0
        return gradient_vector

    def integrate(self, start_x : float, end_x : float):
        left_value = self.interpolate(start_x)
        right_value = self.interpolate(end_x)

        left_index = np.searchsorted(self.axis1_, start_x, side="left")
        right_index = np.searchsorted(self.axis1_, end_x, side="right") - 1
    
        if right_index == -1:
            return (end_x - start_x) * right_value
        if left_index == self.length_:
            return (end_x - start_x) * left_value

        axis = self.axis1_
        values = self.values_
        total_area = ((axis[left_index] - start_x) * left_value + (end_x - axis[right_index]) * right_value)

        for i in range(left_index, right_index):
            segment_width = axis[i + 1] - axis[i]
            total_area += segment_width * values[i + 1]

        return total_area



    def gradient_of_integrated_value_wrt_ordinate(self, start_x : float, end_x : float):
        gradient_vector = np.zeros(self.length_)
        axis = self.axis1_

        left_index = np.searchsorted(axis, start_x, side="left")
        right_index = np.searchsorted(axis, end_x, side="right") - 1

        if left_index == self.length_:
            gradient_vector[-1] = end_x - start_x
            return gradient_vector

        if right_index == -1:
            gradient_vector[0] = end_x - start_x
            return gradient_vector

        gradient_vector[left_index] = axis[left_index] - start_x
        
        for i in range(left_index + 1, right_index + 1):
            gradient_vector[i] += axis[i] - axis[i - 1]

        tail_index = min(right_index + 1, self.length_ - 1)
        gradient_vector[tail_index] += end_x - axis[right_index]
        return gradient_vector

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
