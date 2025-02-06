# Define the number of nodes
N = 32
x = np.cos(np.pi*np.arange(N+1)/N) # Gauss-Lobatto nodes

# source term f(x)

f = lambda x: np.sin(np.pi*x)
F = f(x) # approximate f at x, essentialy compute g_n^* at x

# Chebyshev differentiation matrix
def chebyshev_diff(N,x):
    c = np.ones(N+1)
    c[0] = c[-1] = 2 # similar to LMU Pseudeospectral Methods example
    D = np.zeros((N+1, N+1))
    for i in range(N+1):
        for j in range(N+1):
            if i != j:
                D[i,j] = c[i]*(-1)**(i+j)/(c[j]*(x[i]-x[j]))
        D[i,i] = -np.sum(D[i, :])

    return D
D = chebyshev_diff(N,x)
# Second derivative matrix
D2 = np.dot(D,D)

# Remove rows and columns for Dirichlet boundary conditions
D2_re = D2[1:-1,1:-1]
F_re = F[1:-1]

# Solve the sys
u_re = np.linalg.solve(-D2_re,F_re)

# add boundary conds
u = np.zeros(N+1)
u[1:-1] = u_re

plt.plot(x, u, 'o-', label="Numerical Solution")
plt.plot(x, f(x)/(np.pi**2), '--', label="Exact Solution")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.legend()
plt.title("Chebyshev collocation")
plt.grid()
plt.show()
