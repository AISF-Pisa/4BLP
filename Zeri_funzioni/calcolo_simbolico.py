import sympy as sp

x = sp.Symbol('x')
y = sp.Symbol('y')

z = x/y
print(z)
print(z.subs(x, 4).subs(y, 6))
print(4/6)

#========================================
# Equations
#========================================
print(sp.solve(x**2 - 2, x))
#========================================
# Limit
#========================================
print(sp.limit(sp.log(x)*x, x, 0))
#========================================
# Derivatives
#========================================
f    = sp.sin(x)*sp.exp(x)
dfdx = sp.diff(f, x)
print(dfdx)
#========================================
# Integrals
#========================================
I = sp.integrate(sp.exp(-x**2), (x, -sp.oo, sp.oo))
print(I)
I = sp.integrate(sp.sin(x), x)
print(I)
#========================================
# Differential equations
#========================================
t   = sp.Symbol('t')
y   = sp.Function('y')
eq  = sp.Eq( y(t).diff(t, t) + y(t),  0) # y'' + y = 0
sol = sp.dsolve(eq, y(t))
print(sol)