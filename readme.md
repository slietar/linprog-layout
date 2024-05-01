# linprog-layout

This project is a proof-of-concept layout engine based on linear constraints solved using [`scipy.optimize.linprog()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html).


## Example usage

### System solving

```py
x = Variable('x')
y = Variable('y')

solution = System([
  2 * x + y <= 7,
  x <= 2
]).maximize(3 * x + y)

print(f'x = {x.subs(solution)}')
print(f'y = {y.subs(solution)}')
# => x = 2.0
#    y = 3.0
```

### Layout

```py
img = Image(aspect_ratio=1.2, width=Variable())
gap = HorizontalGap(2)
stack = (gap | img | img | gap).constrain(width=10)

solution = stack.maximize(img.width)

print(f'image width = {img.width.subs(solution)}')
# => image width = 3.0

print(f'image height = {img.height.subs(solution)}')
# => image width = 3.6
```
