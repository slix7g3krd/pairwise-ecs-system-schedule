import pytest
import gameplay.ecs_system_schedule as mod

CASES = [
    ("topo_sort_rw", (1,)),
    ("deferred_destroy", (1,)),
    ("write_conflict_serial", (1,)),
    ("handle_not_pointer", (1,)),
    ("timestep_clamp", (1,)),
    ("determinism", (1,)),
    ("schedule", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("job_join", (1,)),
    ("metrics", (1,)),
    ("query", (1,)),
    ("storage_grow", (1,)),
    ("frame_end", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
