import numpy as np
import scipy.sparse as sps

from potentials import SpringPotential


def state_slices(N):
    return slice(0,N),slice(N,2*N-1),slice(2*N-1,3*N-2),slice(3*N-2,4*N-3)


def split_state(N,x):
    p,r,r_M,z=state_slices(N)
    return x[p],x[r],x[r_M],x[z]


# ----------------------------------------------------------------------
# Geometry and fixed operators
# ----------------------------------------------------------------------

def build_topology(N):
    rows=np.arange(N-1)
    E_L=sps.csr_matrix((np.ones(N-1),(rows,rows)),shape=(N-1,N))
    E_R=sps.csr_matrix((np.ones(N-1),(rows,rows+1)),shape=(N-1,N))
    return E_L,E_R,E_R-E_L


def build_inverse_masses(N,m_light,m_heavy):
    if m_light<=0.0 or m_heavy<=0.0:
        raise ValueError("Masses must be strictly positive.")

    i=np.arange(1,N+1)
    masses=np.where(i%2==1,m_heavy,m_light)
    return 1.0/masses


def matrix_E(N):
    I_N=sps.eye(N,format="csr")
    I=sps.eye(N-1,format="csr")
    Z=sps.csr_matrix((N-1,N-1))
    return sps.block_diag((I_N,I,I,Z),format="csr")


def Q_matrix(N,inverse_masses,K_M,potential: SpringPotential):
    I=sps.eye(N-1,format="csr")
    M_inv=sps.diags(inverse_masses,format="csr")

    return sps.block_diag((
        M_inv,
        potential.linear_stiffness*I,
        K_M*I,
        I,
    ),format="csr")


def B_matrix(N,ell_f=16):
    i=np.arange(1,N+1)
    B_p=np.sin(2.0*np.pi*i/ell_f)
    rows=np.arange(N)

    return sps.csr_matrix(
        (B_p,(rows,np.zeros(N,dtype=int))),
        shape=(4*N-3,1),
    )


# ----------------------------------------------------------------------
# Nonlinear strain coordinates
# ----------------------------------------------------------------------

def physical_strain(r,beta_J):
    r=np.asarray(r,dtype=float)

    if beta_J==0.0:
        return r.copy()

    root=np.sqrt(beta_J)
    return np.arcsinh(root*r)/root


def generalized_strain(delta,beta_J):
    delta=np.asarray(delta,dtype=float)

    if beta_J==0.0:
        return delta.copy()

    root=np.sqrt(beta_J)
    return np.sinh(root*delta)/root


def gamma_vector(r,beta_J):
    return np.sqrt(1.0+beta_J*np.asarray(r,dtype=float)**2)


# ----------------------------------------------------------------------
# Hamiltonian and effort
# ----------------------------------------------------------------------

def Hamiltonian(N,inverse_masses,x,potential: SpringPotential,K_M,beta_J):
    p,r,r_M,_=split_state(N,x)
    delta=physical_strain(r,beta_J)

    kinetic=0.5*np.dot(p,inverse_masses*p)
    primary=np.sum(potential.energy_density(delta))
    maxwell=0.5*K_M*np.dot(r_M,r_M)

    return float(kinetic+primary+maxwell)


def coenergy(N,inverse_masses,x,potential: SpringPotential,K_M,beta_J):
    p,r,r_M,z=split_state(N,x)
    delta=physical_strain(r,beta_J)
    gamma=gamma_vector(r,beta_J)

    return np.concatenate((
        inverse_masses*p,
        potential.gradient(delta)/gamma,
        K_M*r_M,
        z,
    ))


def grad_Hamiltonian(N,inverse_masses,x,potential: SpringPotential,K_M,beta_J):
    p,r,r_M,z=split_state(N,x)
    delta=physical_strain(r,beta_J)
    gamma=gamma_vector(r,beta_J)

    return np.concatenate((
        inverse_masses*p,
        potential.gradient(delta)/gamma,
        K_M*r_M,
        np.zeros_like(z),
    ))


def nonlinear_coenergy(N,x,potential: SpringPotential,beta_J):
    _,r,_,_=split_state(N,x)
    delta=physical_strain(r,beta_J)
    gamma=gamma_vector(r,beta_J)

    e_nl=np.zeros(4*N-3)
    e_nl[N:2*N-1]=(
        potential.gradient(delta)/gamma
        -potential.linear_stiffness*r
    )
    return e_nl


def full_coenergy(Q,x,nonlinear_coenergy):
    return np.asarray(Q@x,dtype=float).ravel()+np.asarray(
        nonlinear_coenergy(x),dtype=float
    ).ravel()


# ----------------------------------------------------------------------
# Interconnection matrices
# ----------------------------------------------------------------------

def _J_from_gamma(N,gamma,topology=None):
    E_L,_,D=build_topology(N) if topology is None else topology
    G=sps.diags(gamma,format="csr")
    I=sps.eye(N-1,format="csr")

    return sps.bmat([
        [None, -D.T@G, E_L.T, None],
        [G@D,  None,   None,  None],
        [-E_L, None,   None,  I],
        [None, None,   -I,    None],
    ],format="csr")


def J0_matrix(N,topology=None):
    return _J_from_gamma(N,np.ones(N-1),topology)


def J_matrix(N,x,beta_J,topology=None):
    _,r,_,_=split_state(N,x)
    return _J_from_gamma(N,gamma_vector(r,beta_J),topology)


# ----------------------------------------------------------------------
# Dissipation matrices
# ----------------------------------------------------------------------

def R0_matrix(N,eta_vector,topology=None):
    eta=np.asarray(eta_vector,dtype=float)

    if eta.shape!=(N-1,) or np.any(eta<=0.0):
        raise ValueError("eta_vector must have shape (N-1,) and be strictly positive.")

    _,E_R,_=build_topology(N) if topology is None else topology
    Lambda=sps.diags(eta,format="csr")
    Z=sps.csr_matrix((N-1,N-1))

    return sps.bmat([
        [E_R.T@Lambda@E_R, None, None, -E_R.T@Lambda],
        [None,             Z,    None, None],
        [None,             None, Z,    None],
        [-Lambda@E_R,      None, None, Lambda],
    ],format="csr")


def R1_matrix(N,inverse_masses,x,beta_R,topology=None):
    _,_,D=build_topology(N) if topology is None else topology
    p,_,_,_=split_state(N,x)

    v=inverse_masses*p
    w=v[1:]-v[:-1]
    Lambda=sps.diags(beta_R*w**2,format="csr")
    Z=sps.csr_matrix((N-1,N-1))

    return sps.block_diag((
        D.T@Lambda@D,
        Z,Z,Z,
    ),format="csr")


def R_matrix(N,inverse_masses,x,eta_vector,beta_R,topology=None):
    return (
        R0_matrix(N,eta_vector,topology)
        +R1_matrix(N,inverse_masses,x,beta_R,topology)
    ).tocsr()


# ----------------------------------------------------------------------
# Constant-core splitting
# ----------------------------------------------------------------------

def linear_A(J0,R0,Q):
    return (-(J0-R0)@Q).tocsr()


def g_nonlinear(N,inverse_masses,x,potential: SpringPotential,beta_J,beta_R):
    """
    Nonlinear remainder

        g = -(J0-R0)e_nl -(J-J0)e + R1 e.

    Evaluated directly without assembling state-dependent matrices.
    """
    p,r,_,_=split_state(N,x)

    v=inverse_masses*p
    w=v[1:]-v[:-1]
    delta=physical_strain(r,beta_J)
    gamma=gamma_vector(r,beta_J)

    link=(
        potential.gradient(delta)
        -potential.linear_stiffness*r
        +beta_R*w**3
    )

    g_p=np.zeros(N)
    g_p[:-1]-=link
    g_p[1:]+=link

    return np.concatenate((
        g_p,
        (1.0-gamma)*w,
        np.zeros(2*(N-1)),
    ))


# ----------------------------------------------------------------------
# Direct nonlinear pH vector field
# ----------------------------------------------------------------------

def vector_field(N,inverse_masses,x,potential: SpringPotential,K_M,
                 eta_vector,beta_J,beta_R):
    """
    f(x) = (J(x)-R(x))e(x).

    Uses the 1D chain structure directly; no sparse J(x) or R(x)
    assembly occurs in the nonlinear residual path.
    """
    p,r,r_M,z=split_state(N,x)

    v=inverse_masses*p
    w=v[1:]-v[:-1]
    delta=physical_strain(r,beta_J)
    gamma=gamma_vector(r,beta_J)

    link=potential.gradient(delta)+beta_R*w**3
    dash=np.asarray(eta_vector)* (v[1:]-z)

    p_dot=np.zeros(N)
    p_dot[:-1]+=link+K_M*r_M
    p_dot[1:]-=link+dash

    return np.concatenate((
        p_dot,
        gamma*w,
        -v[:-1]+z,
        -K_M*r_M+dash,
    ))

def vector_field_jacobian(N,inverse_masses,x,potential: SpringPotential,K_M,
                          eta_vector,beta_J,beta_R,topology=None):
    """Exact sparse Jacobian Df(x) of f(x)=(J(x)-R(x))e(x)."""
    E_L,E_R,D=build_topology(N) if topology is None else topology
    p,r,_,_=split_state(N,x)

    v=inverse_masses*p
    w=v[1:]-v[:-1]
    delta=physical_strain(r,beta_J)
    gamma=gamma_vector(r,beta_J)

    M_inv=sps.diags(inverse_masses,format="csr")
    Lambda=sps.diags(eta_vector,format="csr")
    G=sps.diags(gamma,format="csr")
    I=sps.eye(N-1,format="csr")

    D_M=D@M_inv
    ER_M=E_R@M_inv

    C=sps.diags(potential.curvature(delta)/gamma,format="csr")
    W=sps.diags(3.0*beta_R*w**2,format="csr")
    G_r=sps.diags(beta_J*r*w/gamma,format="csr")

    F_pp=-E_R.T@Lambda@ER_M-D.T@W@D_M
    F_pr=-D.T@C

    return sps.bmat([
        [F_pp,        F_pr, K_M*E_L.T, E_R.T@Lambda],
        [G@D_M,       G_r,  None,       None],
        [-E_L@M_inv,  None, None,       I],
        [Lambda@ER_M, None, -K_M*I,    -Lambda],
    ],format="csr")

def dissipation(N,inverse_masses,x,eta_vector,beta_R):
    p,_,_,z=split_state(N,x)

    v=inverse_masses*p
    w=v[1:]-v[:-1]
    maxwell=v[1:]-z

    return float(
        np.dot(eta_vector,maxwell**2)
        +beta_R*np.dot(w**2,w**2)
    )


# ----------------------------------------------------------------------
# Initial condition
# ----------------------------------------------------------------------

def initial_state(N,a_0,beta_J,ell_0=16):
    i=np.arange(1,N)
    phi=np.sin(2.0*np.pi*i/ell_0)+0.25*np.sin(4.0*np.pi*i/ell_0+np.pi/5.0)
    delta=a_0*phi/np.max(np.abs(phi))
    r=generalized_strain(delta,beta_J)

    return np.concatenate((
        np.zeros(N),
        r,
        np.zeros(N-1),
        np.zeros(N-1),
    ))


# ----------------------------------------------------------------------
# BDF system matrices
# ----------------------------------------------------------------------

def bdf1_system_matrix(E,A,dt):
    if dt<=0.0:
        raise ValueError("dt must be strictly positive.")
    return (E+dt*A).tocsc()


def bdf2_system_matrix(E,A,dt):
    if dt<=0.0:
        raise ValueError("dt must be strictly positive.")
    return (1.5*E+dt*A).tocsc()