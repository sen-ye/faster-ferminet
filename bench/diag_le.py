"""T1 diagnostic: same inputs, override off vs on, bitwise per quantity."""
import os, sys
tree = sys.argv[1]; atom = sys.argv[2]; sys.path.insert(0, tree)
os.environ.setdefault('FERMINET_PALLAS_TRSM', '1')
import jax, jax.numpy as jnp, numpy as np
if jax.default_backend() != 'gpu': raise SystemExit('BACKEND_NOT_GPU: ' + jax.default_backend())
print('backend', jax.default_backend(), jax.devices()[0].device_kind, flush=True)
from ferminet import networks, hamiltonian, envelopes
from ferminet.utils import system
from ferminet.configs import atom as atom_cfg
from ferminet import base_config
from ferminet import pallas_trsm as pt
SPIN = {'Fe': 4, 'Kr': 0, 'Cd': 0, 'Xe': 0}
cfg = atom_cfg.get_config(); cfg.system.atom = atom; cfg.system.spin_polarisation = SPIN[atom]
cfg = base_config.resolve(cfg)
nspins = cfg.system.electrons; B = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
atoms = jnp.stack([jnp.array(a.coords) for a in cfg.system.molecule]); charges = jnp.array([a.charge for a in cfg.system.molecule])
feat = networks.make_ferminet_features(natoms=1, nspins=nspins, ndim=3)
net = networks.make_fermi_net(nspins, charges, determinants=16, envelope=envelopes.make_isotropic_envelope(), feature_layer=feat, jastrow='default', bias_orbitals=False, full_det=True, **cfg.network.ferminet)
params = net.init(jax.random.PRNGKey(0))
ne = sum(nspins)
pos = jax.random.normal(jax.random.PRNGKey(1), (B, 3 * ne)) * 1.0
spins = jnp.tile(jnp.concatenate([jnp.ones(nspins[0]), -jnp.ones(nspins[1])])[None], (B, 1))
data = networks.FermiNetData(positions=pos, spins=spins, atoms=jnp.tile(atoms[None], (B, 1, 1)), charges=jnp.tile(charges[None], (B, 1)))
logabs = lambda p, x, s, a, c: net.apply(p, x, s, a, c)[1]
fns = {
  'logpsi': lambda p, d: jax.vmap(logabs, (None, 0, 0, 0, 0))(p, d.positions, d.spins, d.atoms, d.charges),
  'grad': lambda p, d: jax.vmap(jax.grad(logabs, 1), (None, 0, 0, 0, 0))(p, d.positions, d.spins, d.atoms, d.charges),
  'ke': lambda p, d: jax.vmap(hamiltonian.local_kinetic_energy(net.apply), (None, networks.FermiNetData(positions=0, spins=0, atoms=0, charges=0)))(p, d),
}
# full local energy (kinetic + potential), as used by the loss
_el = hamiltonian.local_energy(net.apply, charges, nspins)
fns['el'] = lambda p, d: jax.vmap(lambda p, k, d: _el(p, k, d)[0], (None, 0, networks.FermiNetData(positions=0, spins=0, atoms=0, charges=0)))(p, jax.random.split(jax.random.PRNGKey(2), B), d)
only = os.environ.get('DIAG_ONLY')
if only: fns = {k: v for k, v in fns.items() if k in only.split(',')}
pt.install()
for name, f in fns.items():
  outs = {}
  for en in (False, True):
    pt.ENABLED = en
    g = jax.jit(lambda p, d, f=f: f(p, d))
    lowered = g.lower(params, data)
    comp = lowered.compile()
    txt = comp.as_text()
    outs[en] = np.asarray(comp(params, data))
    open(f'{sys.argv[4]}/{name}_{"on" if en else "off"}.hlo.txt', 'w').write(txt)
  a, b = outs[False], outs[True]
  mm = int((a.view(np.uint32) != b.view(np.uint32)).sum())
  print(f'{atom} {name}: mismatches {mm}/{a.size} maxabs {np.max(np.abs(a - b)):.3e} pallas_lowerings={pt.STATS}', flush=True)
