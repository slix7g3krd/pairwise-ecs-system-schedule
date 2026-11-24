"""ECS 调度确定性与安全生命周期的行为测试。

仓库自带的契约测试只验证 15 个入口均非 None；这里验证题目描述的六个关键不变量。
"""

from __future__ import annotations

import pytest

import gameplay.ecs_system_schedule as ecs


def test_rw_sets_produce_dataflow_topological_order():
    systems = [
        ecs.System("ReadXWriteY", reads={"X"}, writes={"Y"}),
        ecs.System("ReadYWriteZ", reads={"Y"}, writes={"Z"}),
        ecs.System("WriteX", writes={"X"}),
    ]

    order = ecs.topo_sort_rw(systems)

    assert order == ["WriteX", "ReadXWriteY", "ReadYWriteZ"]
    assert order.stages == [["WriteX"], ["ReadXWriteY"], ["ReadYWriteZ"]]


def test_schedule_fingerprint_and_validation_are_deterministic():
    systems = ecs._demo_systems()

    first = ecs.schedule(systems)
    second = ecs.schedule(list(reversed(systems)))
    report = ecs.determinism(systems)

    # 相同输入重复规划必须完全一致。
    assert first == ecs.schedule(systems)
    assert first.fingerprint == ecs.schedule(systems).fingerprint
    assert report["stable"] is True
    assert report.ok

    validation = ecs.validation(systems)
    assert validation.ok
    assert validation["errors"] == []

    # 反转输入顺序可能改变声明次序推断，但确定性校验本身仍可重复。
    assert second.fingerprint
    assert ecs.validation(systems).ok


def test_deferred_destroy_keeps_components_valid_until_frame_end():
    world = ecs.World()
    entity = world.spawn(ecs.Position(1, 2), ecs.Velocity(3, 4))
    world.destroy(entity)

    # 销毁请求已记录，但帧末统一处理前实体和组件仍然有效。
    assert world.is_destroy_pending(entity)
    assert world.is_alive(entity)
    assert world.query("Position").ids == (entity.id,)
    assert world.get_component(entity, "Position") == ecs.Position(1, 2)

    active = world.query("Position", include_pending=False)
    assert active == []

    report = world.end_frame()

    assert report.destroyed_ids == (entity.id,)
    assert world.query("Position") == []
    assert not world.is_alive(entity)
    with pytest.raises(ecs.StaleHandleError):
        world.get_component(entity, "Position")


def test_deferred_destroy_entrypoint_report_is_safe():
    report = ecs.deferred_destroy(1)

    assert report["component_available_before_flush"] is True
    assert report["released_early"] is False
    assert report["flushed_at_frame_end"] == [1]
    assert 1 not in report["queried_after_flush"]
    assert report["target_present_after_flush"] is False
    assert report["safe"] is True


def test_write_conflicts_are_serialized_into_separate_stages():
    systems = [
        {"name": "WriterA", "writes": {"Transform"}},
        {"name": "Reader", "reads": {"Transform"}},
        {"name": "WriterB", "writes": {"Transform"}},
    ]

    result = ecs.write_conflict_serial(systems)

    assert result.safe is True
    assert result.same_stage_conflicts == []
    assert result.write_groups["Transform"] == ["WriterA", "WriterB"]
    stage_sets = [set(stage) for stage in result.stages]
    for writer in result.write_groups["Transform"]:
        assert sum(writer in stage for stage in stage_sets) == 1
    assert len(result.stages) == 3


def test_handles_survive_storage_growth_and_detect_stale_generation():
    storage = ecs.ComponentStorage("Position", initial_capacity=1)
    handle = ecs.Handle(7, 0)
    storage.set(handle, ecs.Position(9, 9))

    for index in range(100):
        storage.set(ecs.Handle(100 + index, 0), index)

    assert storage.capacity > 1
    assert storage.get(handle) == ecs.Position(9, 9)

    storage.remove(handle)
    with pytest.raises(ecs.StaleHandleError):
        storage.get(handle)

    report = ecs.handle_not_pointer(1)
    assert report["ok"] is True
    assert report["uses_stable_handle"] is True
    assert report["raw_pointer_exposed"] is False


def test_query_snapshot_remains_safe_while_storage_grows():
    world = ecs.World()
    for index in range(10):
        world.spawn(ecs.Position(index, 0))

    query = world.query("Position")
    seen = []
    for index, entity in enumerate(query):
        seen.append(world.get_component(entity, "Position").x)
        if index == 0:
            for extra in range(30):
                world.spawn(ecs.Position(100 + extra, 0))

    assert seen == list(range(10))
    assert len(world.query("Position")) == 40


def test_timestep_is_clamped_and_accumulator_is_bounded():
    result = ecs.timestep_clamp(30.0, max_delta=0.1, fixed_dt=1 / 60, max_steps=5)

    assert float(result) == 0.1
    assert result.steps <= 5
    assert result.accumulator <= (1 / 60) * 5 + 1e-9

    state = ecs.TimeState(max_delta=0.1, fixed_dt=1 / 60, max_steps=5)
    state.pause()
    paused = state.advance(10.0)
    assert float(paused) == 0.0
    assert state.accumulator == 0.0

    state.resume()
    resumed = state.advance(30.0)
    assert float(resumed) == 0.1
    assert resumed.steps <= 5
    assert state.accumulator <= state.max_accumulator + 1e-9


def test_job_join_returns_submission_order_for_any_completion_order():
    jobs = [lambda i=i: i * i for i in range(5)]

    for permutation in [(4, 3, 2, 1, 0), (0, 2, 4, 1, 3), (1, 0, 3, 2, 4)]:
        result = ecs.job_join(jobs, parallel=False, completion_order=permutation)
        assert list(result) == [0, 1, 4, 9, 16]
        assert result.ok
        assert len(result.completion_order) == 5


def test_cycle_is_reported_and_fallback_is_complete_serial_order():
    systems = [
        ecs.System("A", after={"B"}),
        ecs.System("B", after={"A"}),
    ]

    report = ecs.validation(systems)
    fallback = ecs.fallback(systems)

    assert report.ok is False
    assert report["cycle"] is True
    assert fallback.reason == "dependency_cycle"
    assert sorted(fallback) == ["A", "B"]
    assert all(len(stage) == 1 for stage in fallback.stages)


def test_frame_end_entrypoint_flushes_pending_entities():
    world = ecs.World()
    entity = world.spawn(ecs.Position())
    world.destroy(entity)

    report = ecs.frame_end(world)

    assert report.destroyed_ids == (entity.id,)
    assert world.query() == []


def test_storage_grow_entrypoint_preserves_every_handle():
    report = ecs.storage_grow(1)

    assert report["ok"] is True
    assert report["handles_stable"] is True
    assert report["values_preserved"] is True
    assert report["new_capacity"] >= report["old_capacity"]
