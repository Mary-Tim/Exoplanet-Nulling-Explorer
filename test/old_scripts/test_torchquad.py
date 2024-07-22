import time
import torch
from torchquad import Boole, set_up_backend


def example_integrand(x):
    return torch.sum(torch.sin(x), dim=1)


set_up_backend("torch", data_type="float32")
N = 912673
dim = 3
integrator = Boole()
domains = [torch.tensor([[-1.0, y]] * dim) for y in range(10)]

# Integrate without compilation
times_uncompiled = []
for integration_domain in domains:
    t0 = time.perf_counter()
    integrator.integrate(example_integrand, dim, N, integration_domain)
    times_uncompiled.append(time.perf_counter() - t0)

# Integrate with partial compilation
integrate_jit_compiled_parts = integrator.get_jit_compiled_integrate(
    dim, N, backend="torch"
)
times_compiled_parts = []
for integration_domain in domains:
    t0 = time.perf_counter()
    integrate_jit_compiled_parts(example_integrand, integration_domain)
    times_compiled_parts.append(time.perf_counter() - t0)

# Integrate with everything compiled
times_compiled_all = []
integrate_compiled = None
for integration_domain in domains:
    t0 = time.perf_counter()
    if integrate_compiled is None:
        integrate_compiled = torch.jit.trace(
            lambda dom: integrator.integrate(example_integrand, dim, N, dom),
            (integration_domain,),
        )
    integrate_compiled(integration_domain)
    times_compiled_all.append(time.perf_counter() - t0)

print(f"Uncompiled times: {times_uncompiled}")
print(f"Partly compiled times: {times_compiled_parts}")
print(f"All compiled times: {times_compiled_all}")
speedups = [
    (1.0, tu / tcp, tu / tca)
    for tu, tcp, tca in zip(times_uncompiled, times_compiled_parts, times_compiled_all)
]
print(f"Speedup factors: {speedups}")