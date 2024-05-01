import builtins
import functools
import operator
from abc import ABC
from dataclasses import dataclass, field
from pprint import pprint
from typing import Literal, Optional, Self

import numpy as np


_current_anonymous_variable_number = 0

def _generate_anonymous_variable_name():
  global _current_anonymous_variable_number
  _current_anonymous_variable_number += 1
  return '_' + chr(ord('a') + _current_anonymous_variable_number - 1)

@dataclass(eq=False, frozen=True)
class VariableSymbol:
  name: Optional[str] = field(default_factory=_generate_anonymous_variable_name)

  def display(self):
    return self.name or '_'


Solution = dict[VariableSymbol, float]

@dataclass(kw_only=True)
class LinearCombination:
  constant: float
  weights: dict[VariableSymbol, float]

  def __add__(self, other: Self | float):
    match other:
      case LinearCombination():
        return self.__class__(
          constant=(self.constant + other.constant),
          weights={
            var: self.weights.get(var, 0.0) + other.weights.get(var, 0.0) for var in self.weights.keys() | other.weights.keys()
          }
        )

      case builtins.float() | builtins.int():
        return self.__class__(
          constant=(self.constant + other),
          weights=self.weights
        )

      case _:
        return NotImplemented

  def __radd__(self, other: Self | float):
    return self + other

  def __sub__(self, other: Self | float):
    return self + (-other)

  def __rsub__(self, other: Self | float):
    return (-self) + other

  def __mul__(self, other: float):
    return self.__class__(
      constant=(self.constant * other),
      weights={ var: weight * other for var, weight in self.weights.items() }
    )

  def __rmul__(self, other: float):
    return self * other

  def __neg__(self):
    return self * -1.0


  def __le__(self, other: Self | float):
    return InequalityConstraint(self - other)

  def __ge__(self, other: Self | float):
    return InequalityConstraint(other - self)

  def __eq__(self, other: Self | float):
    return EqualityConstraint(other - self)


  def display_partial(self):
    return ''.join(
      ((' - ' if (weight < 0) else ' + ') if (index > 0) else '') +\
      f'{abs(weight) if index > 0 else weight}{var.display()}'
      for index, (var, weight) in enumerate(self.weights.items())
    )

  def subs(self, solution: Solution, /):
    return self.constant + sum(weight * solution[var] for var, weight in self.weights.items())

  def to_array(self, variables: list[VariableSymbol]):
    return np.array([self.weights.get(var, 0.0) for var in variables])

@dataclass
class EqualityConstraint:
  combination: LinearCombination

  def display(self):
    return f'{self.combination.display_partial()} = {-self.combination.constant}'

@dataclass
class InequalityConstraint:
  combination: LinearCombination

  def display(self):
    return f'{self.combination.display_partial()} <= {-self.combination.constant}'

Constraint = EqualityConstraint | InequalityConstraint


def Variable(name: Optional[str] = None):
  return LinearCombination(
    constant=0.0,
    weights={ VariableSymbol(name or _generate_anonymous_variable_name()): 1.0 }
  )

ValueLike = LinearCombination | float

def Value(value: ValueLike):
  match value:
    case LinearCombination():
      return value
    case builtins.float() | builtins.int():
      return LinearCombination(
        constant=value,
        weights={}
      )


@dataclass
class System:
  constraints: list[Constraint]

  def minimize(self, value: LinearCombination, /):
    from scipy.optimize import linprog

    all_vars = list(functools.reduce(operator.or_, (constraint.combination.weights.keys() for constraint in self.constraints)))

    A_all = np.array([constraint.combination.to_array(all_vars) for constraint in self.constraints])
    b_all = np.array([constraint.combination.constant for constraint in self.constraints])
    equality_mask = np.array([isinstance(constraint, EqualityConstraint) for constraint in self.constraints])

    # print(all_vars)
    # print(A_all)
    # print(b_all)
    # print('>', value.to_array(all_vars))

    result = linprog(
      value.to_array(all_vars),
      A_ub=A_all[~equality_mask, :],
      b_ub=-b_all[~equality_mask],
      A_eq=A_all[equality_mask, :],
      b_eq=-b_all[equality_mask]
    )

    if not result.success:
      raise RuntimeError(result.message)

    return { var: value for var, value in zip(all_vars, result.x) }

  def maximize(self, value: LinearCombination, /):
    return self.minimize(-value)


x = Variable('x')
y = Variable('y')
z = Variable('z')

# system = System([
#   x <= 2,
#   x == y + 1,
#   # x <= 12
# ])

# print(system.maximize(x + y))

# print(2 * x + (y + 8) * 5 + x)
# print((8 + y) * 5)
# print(1 - x + y >= 5)


class Surface(ABC):
  def __init__(self):
    self.constraints: list[Constraint]

    self.width: LinearCombination
    self.height: LinearCombination

  def render(self, position: np.ndarray, solution: Solution):
    ...

  def maximize(self, value: LinearCombination):
    return System(self.constraints).maximize(value)

  def minimize(self, value: LinearCombination):
    return System(self.constraints).minimize(value)

  def constrain(
    self,
    *,
    width: Optional[ValueLike] = None,
    height: Optional[ValueLike] = None
  ):
    return ConstrainedSurface(self, width=width, height=height)

  def float(self, *, halign: Literal['center', 'end', 'start']):
    var = Variable('gap')

    match halign:
      case 'start':
        return HorizontalStack([self, HorizontalGap(var)])
      case 'center':
        return HorizontalStack([HorizontalGap(var), self, HorizontalGap(var)])
      case 'end':
        return HorizontalStack([HorizontalGap(var), self])

  def __or__(self, other: 'Surface'):
    return HorizontalStack([self, other])

class ConstrainedSurface(Surface):
  def __init__(self, surface: Surface, *, width: Optional[ValueLike], height: Optional[ValueLike]):
    super().__init__()

    self.width = surface.width
    self.height = surface.height

    self.constraints = surface.constraints.copy()

    if width is not None:
      self.constraints.append(surface.width == Value(width))

    if height is not None:
      self.constraints.append(surface.height == Value(height))

    self._surface = surface

  def render(self, position: np.ndarray, solution: Solution):
    self._surface.render(position, solution)


class Image(Surface):
  def __init__(self, aspect_ratio: float = 1.2, *, width: ValueLike):
    super().__init__()

    max_width = 5
    max_height = 70

    self.width = Value(width)
    self.height = self.width * aspect_ratio

    self.constraints = [
      self.width <= max_width,
      # self.height <= max_height
    ]

  def render(self, position: np.ndarray, solution: Solution):
    print(f'Rendering image at {position}')

class HorizontalStack(Surface):
  def __init__(self, surfaces: list[Surface]):
    super().__init__()

    assert surfaces

    self.width = sum((surface.width for surface in surfaces), 0.0) # type: ignore
    self.height = surfaces[0].height

    self.constraints = [constraint for surface in surfaces for constraint in surface.constraints] +\
      [surface.height == self.height for surface in surfaces[1:]]

    self._surfaces = surfaces

  def render(self, position: np.ndarray, solution: Solution):
    offset = 0.0

    for surface in self._surfaces:
      surface.render(position + np.array([offset, 0.0]), solution)
      offset += surface.width.subs(solution)

  def __or__(self, other: Surface):
    return HorizontalStack([*self._surfaces, other])

class HorizontalGap(Surface):
  def __init__(self, size: ValueLike):
    super().__init__()

    self.width = Value(size)
    self.height = Variable()

    self.constraints = []

  def render(self, position: np.ndarray, solution: Solution):
    pass


# img = Image(width=x)
# # fr = Variable('fr')
# # stack = HorizontalStack([img, HorizontalGap(fr), img]).constrain(height=7.3)
# # stack = HorizontalStack([img, HorizontalGap(fr)]).constrain(width=6)

# # stack = img.float(halign='center').constrain(width=100)
# stack = (img | img | img | img)

# # pprint(stack.constraints)
# # pprint(img.constraints)

# # for constraint in stack.constraints:
# #   print(constraint.display())

# # print(stack.height)
# sol = stack.maximize(img.width)
# print(sol)

# stack.render(np.array([0.0, 0.0]), sol)


img = Image(aspect_ratio=1.2, width=Variable())
gap = HorizontalGap(2)
stack = (gap | img | img | gap).constrain(width=10)

# print(stack._surface._surfaces)

# for constraint in stack.constraints:
#   print(constraint.display())

solution = stack.maximize(img.width)

# print(x + 2 == y)
# print('>>', x + 2.0 == y)

print(f'image width = {img.width.subs(solution)}')
print(f'image height = {img.height.subs(solution)}')

stack.render(np.array([0.0, 0.0]), solution)
