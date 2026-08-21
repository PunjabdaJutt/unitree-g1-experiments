# Port-Hamiltonian Pipeline for Unitree G1 in IsaacLab

This document is a **step-by-step implementation guide** for building the full port-Hamiltonian (PH) and control pipeline for the Unitree G1 from scratch, based on the G1 PH process document and the PH multibody dynamics paper.[file:4][file:13]

It is organized by phases. For each phase you get:
- Conceptual goal
- Data structures
- Algorithms
- Suggested Python modules and function signatures
- Testing/validation strategy

You can mirror these sections into your `src/` layout (e.g. `ph/urdf.py`, `ph/tree.py`, `ph/system.py`, `ph/controller.py`, `ph/integrators.py`).

---

## Phase 1: URDF Parsing and Robot Model

### Goal

Extract the full mechanical description of the Unitree G1 (including hands) from URDF/XACRO into a structured `RobotModel` that feeds all later stages.[file:4]

### Key concepts

- Links: mass, COM position, inertia tensor.
- Joints: type (revolute/fixed), axis vector, parent–child transform (RPY + xyz), joint limits.[file:4]
- G1 structure: 63 links, 62 joints, 53 movable joints.[file:4]

### Data structures

Create a module `ph/urdf_model.py` with:

```python
@dataclass
class Link:
    name: str
    mass: float
    com: np.ndarray        # shape (3,)
    inertia: np.ndarray    # shape (3, 3) in link frame

@dataclass
class Joint:
    name: str
    type: str              # 'revolute', 'fixed', etc.
    parent: str
    child: str
    axis: np.ndarray       # shape (3,)
    origin_xyz: np.ndarray # parent->joint translation
    origin_rpy: np.ndarray # parent->joint RPY
    limit_lower: float
    limit_upper: float

@dataclass
class RobotModel:
    links: Dict[str, Link]
    joints: Dict[str, Joint]
    root_link: str
```

### Algorithms

Implement `RobotModel.from_urdf(path: str) -> RobotModel`:

1. Use `urdfpy` or your own XML parser to read the URDF/XACRO.
2. For each `<link>`:
   - Read `<inertial>` mass, origin, and inertia tensor.[file:4]
   - Store as `Link`.
3. For each `<joint>`:
   - Read type, parent, child.[file:4]
   - Read `<origin xyz="..." rpy="..."/>`.
   - Read `<axis xyz="..."/>`.
   - Read `<limit lower="..." upper="..."/>`.
   - Store as `Joint`.
4. Determine `root_link` as the link that is only a parent and never a child.

### Suggested Python file

- `ph/urdf_parser.py` implementing `RobotModel.from_urdf`.

### Tests & validation

- Check that the number of links and joints matches the G1 URDF (63 links, 62 joints, 53 movable).[file:4]
- Verify that COM and inertia tensors are loaded correctly for a few sample links.
- Confirm that joint axes and RPY transforms match URDF for critical joints (pelvis, shoulders, hands).

---

## Phase 2: Spatial Algebra and Multibody Tree (Featherstone)

### Goal

Build a Featherstone-style multibody tree from the URDF model so you can compute dynamics efficiently (mass matrix, inverse dynamics).[file:4]

### Key concepts

- Spatial motion vector `v` (angular first, linear second).[file:4]
- Spatial force vector `f`.
- Spatial transforms:
  - `Xrot(E)` rotation-only.[file:4]
  - `Xtrans(r)` translation-only.[file:4]
  - `XJ(axis, q)` joint transform for revolute joints.[file:4]
- Spatial inertia `I_spatial(m, c, I_com)`.[file:4]
- Tree of bodies built by breadth-first walk from pelvis.

### Data structures

Create `ph/tree.py` with:

```python
@dataclass
class Body:
    name: str
    parent: int
    Xtree: np.ndarray   # shape (6, 6)
    I: np.ndarray       # spatial inertia (6, 6)
    S: np.ndarray       # motion subspace (6, 1)

@dataclass
class MultiBodyTree:
    bodies: List[Body]
    joint_names: List[str]
    joint_to_body: Dict[str, int]
```

### Algorithms

Implement `build_tree_from_urdf(model: RobotModel) -> MultiBodyTree`:

1. Start from pelvis as fixed base.[file:4]
2. Perform breadth-first traversal of the joint graph.
3. For each **movable joint** encountered:
   - Define a new `Body`.
   - Compute `Xtree[i]` as the transform from parent body frame to joint pre-frame using `origin_rpy` and `origin_xyz`.
   - Compute spatial inertia `I[i]` by aggregating all links rigidly attached through fixed joints (sum of their spatial inertias).[file:4]
   - Define motion subspace `S[i]` for revolute joint as `S = [axis; 0]` (angular only).[file:4]
4. For each **fixed joint**:
   - Do **not** create a new body.
   - Accumulate its transform into the current body’s frame.
5. Store `parent` indices and `joint_names`.

#### Spatial operators

Create helper functions:

```python
def Xrot(E: np.ndarray) -> np.ndarray:
    # 6x6 rotation-only transform
    # [E 0; 0 E]

def Xtrans(r: np.ndarray) -> np.ndarray:
    # [I 0; skew(r) I]

def I_spatial(m: float, c: np.ndarray, I_com: np.ndarray) -> np.ndarray:
    # spatial inertia matrix from mass, COM, and inertia tensor

def XJ_revolute(axis: np.ndarray, q: float) -> np.ndarray:
    # joint transform as rotation about axis by q using Rodrigues formula
```

### Suggested Python file

- `ph/spatial_algebra.py` for operators.
- `ph/tree.py` for `MultiBodyTree`.

### Tests & validation

- Check that the resulting tree has 53 bodies for the G1.[file:4]
- Verify `Xtree` and `I` for a small chain (e.g. pelvis to left hand) against manual computations.
- Compare FK using this tree to IsaacSim’s physics inspector poses for several joint configurations.[file:3]

---

## Phase 3: Mass Matrix via CRBA and Inverse Dynamics

### Goal

Compute the joint-space mass matrix `M(q)` and inverse dynamics torques efficiently using Featherstone’s Composite Rigid Body Algorithm (CRBA) and RNEA.[file:4]

### Key concepts

- Composite inertia `IC[i]` for each subtree.
- Upward transforms `Xup[i]` combining joint motion and tree transforms.[file:4]
- CRBA two-pass algorithm.

### Algorithms

Implement `compute_mass_matrix(tree: MultiBodyTree, q: np.ndarray) -> np.ndarray`:

**Pass 1 (leaf to root):**[file:4]

1. For each body `i`, compute `XJ[i] = XJ_revolute(axis_i, q_i)`.
2. Compute `Xup[i] = XJ[i] @ Xtree[i]`.
3. Initialize `IC[i] = I[i]`.
4. Traverse bodies from leaves to root:
   - For child `i`, accumulate into parent `p`:
     - `IC[p] = IC[p] + Xup[i].T @ IC[i] @ Xup[i]`.

**Pass 2 (fill M):**[file:4]

1. Initialize `M` as zero matrix of size (n_joints, n_joints).
2. For each joint `i`:
   - Compute `F = IC[i] @ S[i]`.
   - `M[i, i] = S[i].T @ F`.
   - Walk up the tree from `j = parent[i]` while `j != -1`:
     - `F = Xup[j].T @ F`.
     - `M[i, j] = S[j].T @ F`.
     - `M[j, i] = M[i, j]` (symmetry).

Implement `inverse_dynamics(tree, q, dq, ddq) -> tau` using RNEA (standard algorithm from Featherstone).[file:4]

### Suggested Python file

- `ph/dynamics.py` containing CRBA and RNEA.

### Tests & validation

- Confirm `M(q)` is symmetric and positive definite for random valid `q`.[file:4]
- Compare RNEA torques against a simple finite-difference check or another dynamics library.

---

## Phase 4: Port-Hamiltonian State-Space Construction

### Goal

Build the PH state-space representation `x = (q, p)`, Hamiltonian `H(q, p)`, and PH structure matrices `J`, `R`, `G` for the G1 multibody tree.[file:4]

### Key concepts

- State `x ∈ R^{2n}`: `q ∈ R^n`, `p = M(q) dq`.[file:4]
- Hamiltonian `H = T + U` with kinetic `T = 0.5 pᵀ M⁻¹ p` and potential `U(q)` from gravity.[file:4]
- Dirac interconnection matrix `J` with canonical symplectic structure.[file:4]
- Dissipation matrix `R` (joint damping).
- Control port matrix `G` selecting HAND12 joints.[file:4]

### Data structures

Create `ph/system.py` with:

```python
@dataclass
class PHSystem:
    tree: MultiBodyTree
    hand12_indices: np.ndarray  # indices of HAND12 joints
    damping: np.ndarray         # diag(d0, ..., d_{n-1})
```

### Algorithms

Implement:

```python
def ph_state(tree: MultiBodyTree, q: np.ndarray, dq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    M = compute_mass_matrix(tree, q)
    p = M @ dq
    return q, p


def kinetic_energy(tree: MultiBodyTree, q: np.ndarray, p: np.ndarray) -> float:
    M = compute_mass_matrix(tree, q)
    # Solve M v = p
    v = np.linalg.solve(M, p)
    return 0.5 * float(p @ v)


def potential_energy(tree: MultiBodyTree, q: np.ndarray, link_masses: Dict[int, float], com_world: Dict[int, np.ndarray]) -> float:
    # Walk FK tree, compute each body's COM in world coordinates and sum m_i g z_i.[file:4]


def hamiltonian(tree, q, p) -> float:
    T = kinetic_energy(tree, q, p)
    U = potential_energy(tree, q, p)
    return T + U
```

Structure matrices:

```python
def J_matrix(n: int) -> np.ndarray:
    # J = [[0, I], [-I, 0]]
    J = np.zeros((2*n, 2*n))
    J[0:n, n:2*n] = np.eye(n)
    J[n:2*n, 0:n] = -np.eye(n)
    return J


def R_matrix(damping: np.ndarray) -> np.ndarray:
    # R acting only on velocities (q dot)
    n = damping.shape[0]
    R = np.zeros((2*n, 2*n))
    R[n:2*n, n:2*n] = np.diag(damping)
    return R


def G_matrix(n: int, hand12_indices: np.ndarray) -> np.ndarray:
    # control port matrix selecting HAND12 joints
    B = np.zeros((n, hand12_indices.shape[0]))
    for col, idx in enumerate(hand12_indices):
        B[idx, col] = 1.0
    G = np.zeros((2*n, hand12_indices.shape[0]))
    G[n:2*n, :] = B
    return G
```

PH dynamics:[file:4]

```python
def ph_rhs(tree: MultiBodyTree, q: np.ndarray, p: np.ndarray, damping: np.ndarray,
           hand12_indices: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = q.shape[0]
    J = J_matrix(n)
    R = R_matrix(damping)
    G = G_matrix(n, hand12_indices)

    # gradients
    # p-gradient: ∂H/∂p = v = M^{-1} p
    M = compute_mass_matrix(tree, q)
    v = np.linalg.solve(M, p)
    grad_p = v

    # q-gradient via central finite differences as in the G1 PH process doc.[file:4]
    grad_q = finite_diff_grad_q(tree, q, p)

    grad = np.concatenate([grad_q, grad_p])

    x = np.concatenate([q, p])
    dx = (J - R) @ grad + G @ u
    dq_dt = dx[:n]
    dp_dt = dx[n:]
    return dq_dt, dp_dt
```

Implement `finite_diff_grad_q(tree, q, p)` exactly as described in the PH process document: two CRBA evaluations per coordinate, central differences.[file:4]

### Tests & validation

- Verify energy conservation (no damping, no control) using PH-MP integrator (next phase).[file:13]
- Check that `J` is skew-symmetric and `R` symmetric positive semidefinite.[file:4]
- Confirm that HAND12 indices select the intended proximal finger joints.[file:4]

---

## Phase 5: Finger Controller and Gravity Compensation

### Goal

Design and implement an IDA-PBC-style finger controller with gravity compensation for the HAND12 joints, driven by PH dynamics but connected to the simulator state.[file:4]

### Control law

As given in the process document:[file:4]

\[ u = g_c(q) + K_p (q_c - q) + K_d (\dot q_c - \dot q) + K_i z \]

- `g_c(q) = U_q(q)_{HAND12}`: gravity compensation from PH potential gradient on HAND12 indices.[file:4]
- `q_c`: desired finger joint positions.
- `z`: integral state evolved as `\dot z = e` with clamping.

### Algorithms

Implement in `ph/controller.py`:

```python
@dataclass
class FingerController:
    ph_system: PHSystem
    Kp: np.ndarray  # HAND12 x HAND12
    Kd: np.ndarray
    Ki: np.ndarray
    z: np.ndarray   # integral state HAND12
    z_limits: Tuple[float, float]

    def gravity_compensation(self, q: np.ndarray) -> np.ndarray:
        # compute grad_q on full tree, restrict to HAND12
        grad_q_full = finite_diff_grad_q(self.ph_system.tree, q, np.zeros_like(q))
        return grad_q_full[self.ph_system.hand12_indices]

    def step(self, q_sim: np.ndarray, dq_sim: np.ndarray, q_des_hand: np.ndarray,
             dt: float) -> np.ndarray:
        # 1) Lift simulator q, dq to tree state (Phase 6 mapping).
        q_tree, dq_tree = lift_sim_to_tree(q_sim, dq_sim)

        # 2) Extract current HAND12 joint states.
        q_hand = q_tree[self.ph_system.hand12_indices]
        dq_hand = dq_tree[self.ph_system.hand12_indices]

        # 3) Error and gravity compensation.
        e = q_des_hand - q_hand
        gc = self.gravity_compensation(q_tree)

        # 4) Update integral state with clamping.
        self.z += e * dt
        self.z = np.clip(self.z, self.z_limits[0], self.z_limits[1])

        # 5) Control law.
        u = gc + self.Kp @ e + self.Kd @ (-dq_hand) + self.Ki @ self.z
        return u
```

### Tests & validation

- In quasi-static scenarios (slow motion, `p ≈ 0`), verify that gravity compensation matches the HAND12 entries of `∂H/∂q`.[file:4]
- Compare torque tracking performance to the closed-loop torque controller experiments in the G1 manipulation report.[file:3]

---

## Phase 6: Simulator–Tree Index Mapping (IsaacLab)

### Goal

Map IsaacSim/IsaacLab’s 29-joint articulation state to the 53-DOF PH tree state and back, so controllers and PH dynamics can operate correctly.[file:4]

### Data structures

Create `ph/mapping.py`:

```python
@dataclass
class SimTreeMap:
    sim_joint_names: List[str]
    tree_joint_names: List[str]
    sim_to_tree: Dict[int, int]
    tree_to_sim: Dict[int, int]
```

### Algorithms

1. Build mapping at initialization:
   - For each simulator joint name, find matching tree joint name.
   - Store index pairs.

2. Implement lift:

```python
def lift_sim_to_tree(map: SimTreeMap, q_sim: np.ndarray, dq_sim: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n_tree = len(map.tree_joint_names)
    q_tree = np.zeros(n_tree)
    dq_tree = np.zeros(n_tree)
    for sim_idx, tree_idx in map.sim_to_tree.items():
        q_tree[tree_idx] = q_sim[sim_idx]
        dq_tree[tree_idx] = dq_sim[sim_idx]
    return q_tree, dq_tree
```

3. Implement restrict:

```python
def scatter_hand12_to_sim(map: SimTreeMap, hand12_indices: np.ndarray, u_hand12: np.ndarray) -> np.ndarray:
    tau_sim = np.zeros(len(map.sim_joint_names))
    for col, tree_idx in enumerate(hand12_indices):
        sim_idx = map.tree_to_sim[tree_idx]
        tau_sim[sim_idx] = u_hand12[col]
    return tau_sim
```

### Tests & validation

- Confirm that for a known configuration the lifted `q_tree` reproduces simulator poses through FK.[file:4]
- Ensure HAND12 indices correctly map to the six actuated joints on each hand described in the URDF and process doc.[file:4]

---

## Phase 7: Structure-Preserving Integrator (PH-MP / PH-MP-GGL)

### Goal

Implement a structure-preserving integrator (implicit midpoint PH-MP and PH-MP-GGL) for the PH-DAE system, following the multibody PH paper.[file:13]

### Key concepts

- Midpoint rule on PH-DAE: `E x_{n+1} - x_n = h J(x_{n+1/2}) z(x_{n+1/2}) + h B(x_{n+1/2}) u_{n+1/2}`.[file:13]
- Exact energy and angular momentum conservation for quadratic Hamiltonians and director-based rigid bodies.[file:13]
- GGL extension for enforcing velocity-level constraints.[file:13]

### Algorithms

In `ph/integrators.py`, implement:

```python
def ph_mp_step(ph_system: PHSystem, x_n: np.ndarray, u_mid: np.ndarray, h: float) -> np.ndarray:
    # x = [q; p] or extended including constraints
    # Use Newton iteration to solve implicit midpoint equation.

    def residual(x_np1: np.ndarray) -> np.ndarray:
        x_mid = 0.5 * (x_np1 + x_n)
        # unpack
        q_mid = x_mid[:n]
        p_mid = x_mid[n:2*n]
        dq_mid, dp_mid = ph_rhs(ph_system.tree, q_mid, p_mid,
                                ph_system.damping, ph_system.hand12_indices, u_mid)
        rhs = np.concatenate([dq_mid, dp_mid])
        return ph_system.E @ (x_np1 - x_n) - h * rhs

    # Newton solve for x_{n+1}
    x_np1 = newton_solve(residual, x_n)
    return x_np1
```

For PH-MP-GGL, extend `x` and `ph_rhs` with velocity-level constraints and Lagrange multipliers as in the PH-DAE formulation (equation (61) in the multibody paper).[file:13]

### Tests & validation

- Reproduce the flying cylindrical pair and closed-loop benchmarks, checking energy and angular momentum conservation.[file:13]
- Confirm constraint satisfaction on position level (PH-MP) and velocity level (PH-MP-GGL).[file:13]
- Compare slider-crank displacement and velocity profiles to reference results (Rosenbrock, Lie group methods).[file:13]

---

## Phase 8: High-Level IsaacLab Task Scripts

### Goal

Combine all phases into IsaacLab tasks (e.g. torque-controlled grasping, follow box, pick–lift–place) for the G1.[file:3][file:4]

### Task loop template

In `tasks/g1_finger_ph_grasp.py`:

```python
def run_episode(env, ph_system: PHSystem, controller: FingerController, map: SimTreeMap,
                q_des_hand_traj: np.ndarray, T: float, dt: float):
    t = 0.0
    while t < T:
        # 1) Read simulator joint states.
        q_sim = env.get_joint_positions()
        dq_sim = env.get_joint_velocities()

        # 2) Get desired hand trajectory at time t.
        q_des_hand = q_des_hand_traj(t)

        # 3) Compute HAND12 torques.
        u_hand12 = controller.step(q_sim, dq_sim, q_des_hand, dt)

        # 4) Scatter to simulator.
        tau_sim = scatter_hand12_to_sim(map, ph_system.hand12_indices, u_hand12)

        # 5) Apply torques and step simulation.
        env.set_joint_efforts(tau_sim)
        env.step(dt)

        # 6) Log q_sim, dq_sim, tau_sim, energies.
        t += dt
```

### Tests & validation

- Validate that grasp torque tracks desired values under varying contact conditions (compare to force-impedance mapping).[file:3]
- Check that full pipeline (URDF → tree → PH → controller → mapping → simulator) behaves consistently and is numerically stable.

---

## Phase 9: Benchmark & Paper Reproduction

### Goal

Use your codebase to reproduce key figures and tables from:
- G1 torque-controlled grasping report (box tracking, grasp torque comparison).[file:3]
- PH multibody dynamics paper (energy, constraints, slider-crank).[file:13]

### Suggested workflow

1. Create `experiments/` scripts that:
   - Run PH integrators on benchmark systems (cylindrical pair, closed loop, slider-crank).[file:13]
   - Run IsaacLab episodes with G1 grasping tasks.[file:3][file:4]
2. Use `ph_experiment_logger` to log trajectories.
3. Use plotting scripts to reproduce energy, angular momentum, constraint error, and displacement/velocity plots.

This completes a from-scratch, phase-structured guide for building your PH and control codebase around the Unitree G1 and IsaacLab.
