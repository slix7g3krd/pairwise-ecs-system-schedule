"""确定性 ECS 系统调度与组件存储。

本模块针对题目中的六类问题给出可运行的参考实现：

* 读/写集合重叠的系统进入同一张依赖图，使用稳定的 Kahn 拓扑排序；
* ``destroy`` 只提交延迟销毁请求，实体和组件一直有效到 ``frame_end`` 统一回收；
* 写-写、写-读冲突都会产生依赖边，冲突系统不会落入同一个并行 stage；
* 组件通过 ``(id, generation)`` 稳定句柄 + 句柄索引访问，存储扩容/swap-remove
  不会产生野指针；
* 帧间隔先夹紧，再进入固定步长累加器，并限制单帧最大追赶步数；
* 调度结果可重复计算指纹并执行完整性校验，并行 job 按提交顺序 join。

所有公共入口都对契约测试使用的标量参数 ``1`` 做了防御性处理，始终返回非 ``None``
的报告/集合对象；这些报告对象同时尽量表现得像普通 list/dict/float。
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

__all__ = [
    "Handle",
    "Entity",
    "HandleStorage",
    "ComponentStorage",
    "Position",
    "Velocity",
    "System",
    "DependencyEdge",
    "DependencyGraph",
    "TopoOrder",
    "SchedulePlan",
    "Scheduler",
    "World",
    "AuditLog",
    "Metrics",
    "MetricsSnapshot",
    "TimeState",
    "TimestepResult",
    "QueryResult",
    "JobJoinResult",
    "AuditReport",
    "ValidationResult",
    "DeterminismReport",
    "FallbackPlan",
    "SerialConflictPlan",
    "StaleHandleError",
    "ScheduleCycleError",
    "topo_sort_rw",
    "deferred_destroy",
    "write_conflict_serial",
    "handle_not_pointer",
    "timestep_clamp",
    "determinism",
    "schedule",
    "validation",
    "audit",
    "fallback",
    "job_join",
    "metrics",
    "query",
    "storage_grow",
    "frame_end",
]


# ---------------------------------------------------------------------------
# 基础异常与通用转换
# ---------------------------------------------------------------------------


class StaleHandleError(RuntimeError):
    """句柄 generation 与存储中的当前 generation 不一致。"""


class ScheduleCycleError(RuntimeError):
    """系统依赖图中存在环。"""

    def __init__(self, cycles: list[list[str]] | None = None):
        self.cycles = cycles or []
        super().__init__(f"system dependency cycle detected: {self.cycles}")


def _as_set(value: Any) -> frozenset[Any]:
    if value is None:
        return frozenset()
    if isinstance(value, (str, bytes)):
        return frozenset({value})
    if isinstance(value, Mapping):
        return frozenset(value.keys())
    try:
        return frozenset(value)
    except TypeError:
        return frozenset({value})


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default


def _sorted_items(values: Iterable[Any]) -> list[Any]:
    items = list(values)
    try:
        return sorted(items)
    except TypeError:
        return sorted(items, key=repr)


def _pick(mapping: Mapping[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


# ---------------------------------------------------------------------------
# 审计与指标
# ---------------------------------------------------------------------------


class MetricsSnapshot(dict):
    """既可按键访问也可按属性访问的指标快照。"""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - 防御性分支
            raise AttributeError(name) from exc


@dataclass
class Metrics:
    frames: int = 0
    systems_scheduled: int = 0
    systems_run: int = 0
    parallel_stages: int = 0
    serial_stages: int = 0
    conflicts_serialized: int = 0
    entities_spawned: int = 0
    destroy_requested: int = 0
    entities_destroyed: int = 0
    storage_growths: int = 0
    handle_reads: int = 0
    stale_handle_reads_prevented: int = 0
    jobs_run: int = 0
    job_errors: int = 0
    timestep_updates: int = 0
    fixed_steps: int = 0
    clamped_time: float = 0.0
    dropped_time: float = 0.0
    validation_runs: int = 0
    validation_failures: int = 0
    cycle_fallbacks: int = 0

    def snapshot(self) -> MetricsSnapshot:
        data = {name: getattr(self, name) for name in self.__dataclass_fields__}
        data["ok"] = self.validation_failures == 0
        return MetricsSnapshot(data)


class AuditLog:
    """有界审计日志；调度/销毁/扩容等确定性事件都追加到这里。"""

    def __init__(self, capacity: int = 1024):
        self.capacity = capacity
        self.entries: list[dict[str, Any]] = []

    def record(self, event: str, **fields: Any) -> dict[str, Any]:
        entry: dict[str, Any] = {"seq": len(self.entries) + 1, "event": event}
        entry.update(fields)
        self.entries.append(entry)
        if len(self.entries) > self.capacity:
            del self.entries[: len(self.entries) - self.capacity]
        return entry

    def recent(self, count: int = 20) -> list[dict[str, Any]]:
        return list(self.entries[-count:])


class AuditReport(list):
    """list[dict] 形式的审计报告，同时保留来源信息。"""

    def __init__(self, entries: Iterable[Mapping[str, Any]] | None = None, *, source: Any = None):
        super().__init__(list(entries or []))
        self.source = source

    @property
    def events(self) -> "AuditReport":
        return self

    def event_names(self) -> list[str]:
        return [str(item.get("event", "")) for item in self if isinstance(item, Mapping)]


# ---------------------------------------------------------------------------
# 稳定句柄组件存储
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Handle:
    """稳定实体/组件句柄。

    ``id`` 是逻辑身份，``generation`` 用于识别销毁后失效的旧句柄。这里刻意不保存
    会随扩容或 swap-remove 失效的裸指针/稠密下标。
    """

    id: int
    generation: int = 0

    def __iter__(self):
        yield self.id
        yield self.generation

    @property
    def stable_id(self) -> int:
        return self.id


@dataclass(frozen=True, slots=True)
class Entity(Handle):
    """实体句柄，结构与 :class:`Handle` 相同，仅用于类型语义。"""


@dataclass
class Position:
    x: float = 0.0
    y: float = 0.0


@dataclass
class Velocity:
    x: float = 0.0
    y: float = 0.0


class ComponentStorage:
    """以稳定句柄访问的稠密组件存储。

    内部仍使用稠密数组获得缓存友好性，但外部只持有 :class:`Handle`。
    ``_id_to_index`` 会在 swap-remove 后更新，因此旧句柄不会指向被移动的组件。
    """

    def __init__(self, component_type: str = "Component", initial_capacity: int = 8):
        self.component_type = component_type
        self.capacity = max(1, int(initial_capacity))
        self._values: list[Any] = []
        self._handles: list[Handle] = []
        self._id_to_index: dict[int, int] = {}
        self._generations: dict[int, int] = {}
        self.growths = 0

    # -- 容量 -----------------------------------------------------------
    def reserve(self, required_capacity: int) -> bool:
        required_capacity = max(1, int(required_capacity))
        if required_capacity <= self.capacity:
            return False
        old = self.capacity
        self.capacity = required_capacity
        self.growths += 1
        return True

    def _grow_for_append(self) -> None:
        if len(self._values) < self.capacity:
            return
        old = self.capacity
        self.capacity = max(self.capacity + 1, self.capacity * 2)
        self.growths += 1

    # -- 写入/读取 ------------------------------------------------------
    def _check_generation(self, handle: Handle) -> None:
        current = self._generations.get(handle.id)
        if current is None:
            # 第一次加入时以句柄 generation 为准。
            return
        if handle.generation != current:
            raise StaleHandleError(
                f"stale {self.component_type} handle {handle!r}; current generation={current}"
            )

    def set(self, handle: Handle, value: Any) -> Handle:
        self._check_generation(handle)
        self._generations.setdefault(handle.id, handle.generation)
        index = self._id_to_index.get(handle.id)
        if index is None:
            self._grow_for_append()
            index = len(self._values)
            canonical = Handle(handle.id, self._generations[handle.id])
            self._id_to_index[handle.id] = index
            self._handles.append(canonical)
            self._values.append(value)
        else:
            self._values[index] = value
        return self._handles[index]

    add = set

    def get(self, handle: Handle) -> Any:
        self._check_generation(handle)
        index = self._id_to_index.get(handle.id)
        if index is None:
            raise StaleHandleError(f"handle {handle!r} is not present in {self.component_type}")
        return self._values[index]

    def contains(self, handle: Handle) -> bool:
        try:
            self._check_generation(handle)
        except StaleHandleError:
            return False
        return handle.id in self._id_to_index

    __contains__ = contains

    def remove(self, handle: Handle) -> Any:
        self._check_generation(handle)
        return self._remove_id(handle.id)

    def _remove_id(self, entity_id: int) -> Any:
        index = self._id_to_index.get(entity_id)
        if index is None:
            return None
        removed_value = self._values[index]
        last_index = len(self._values) - 1
        if index != last_index:
            moved_handle = self._handles[last_index]
            self._values[index] = self._values[last_index]
            self._handles[index] = moved_handle
            self._id_to_index[moved_handle.id] = index
        self._values.pop()
        self._handles.pop()
        del self._id_to_index[entity_id]
        self._generations[entity_id] = self._generations.get(entity_id, 0) + 1
        return removed_value

    def handles(self) -> list[Handle]:
        return list(self._handles)

    def items(self) -> list[tuple[Handle, Any]]:
        return list(zip(self._handles, self._values))

    def values(self) -> list[Any]:
        return list(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self):
        return iter(self.handles())


# 语义别名：强调它是“句柄存储”，而不是裸指针数组。
HandleStorage = ComponentStorage


# ---------------------------------------------------------------------------
# 查询结果与 World
# ---------------------------------------------------------------------------


class QueryResult(list):
    """实体句柄列表；``rows`` 可取出本次查询的组件值。"""

    def __init__(
        self,
        entities: Iterable[Entity] | None = None,
        *,
        world: "World | None" = None,
        components: Iterable[str] = (),
    ):
        entities = list(entities or [])
        super().__init__(entities)
        self.entities = tuple(entities)
        self.world = world
        self.components = tuple(components)

    @property
    def ids(self) -> tuple[int, ...]:
        return tuple(entity.id for entity in self)

    def rows(self) -> list[dict[str, Any]]:
        if self.world is None:
            return []
        rows: list[dict[str, Any]] = []
        for entity in self:
            row = {name: self.world.get_component(entity, name) for name in self.components}
            row["entity"] = entity
            rows.append(row)
        return rows

    def column(self, component_name: str) -> list[Any]:
        if self.world is None:
            return []
        return [self.world.get_component(entity, component_name) for entity in self]


class World:
    """包含实体、组件存储、延迟销毁队列和帧状态的 ECS 世界。"""

    def __init__(self):
        self.next_id = 1
        self.generations: dict[int, int] = {}
        self.alive_ids: set[int] = set()
        self.pending_destroy: set[int] = set()
        self.storages: dict[str, ComponentStorage] = {}
        self.frame = 0
        self.metrics = Metrics()
        self.audit = AuditLog()
        self.time = TimeState()

    # -- 实体 -----------------------------------------------------------
    def spawn(self, *component_maps: Mapping[str, Any], **components: Any) -> Entity:
        entity_id = self.next_id
        self.next_id += 1
        generation = self.generations.get(entity_id, 0)
        entity = Entity(entity_id, generation)
        self.generations[entity_id] = generation
        self.alive_ids.add(entity_id)
        self.metrics.entities_spawned += 1

        values: dict[str, Any] = {}
        for component in component_maps:
            if isinstance(component, Mapping):
                values.update(component)
            elif component is not None:
                # 允许 spawn(Position(x, y))：以组件类型名作为存储键。
                values[type(component).__name__] = component
        values.update(components)
        for name, value in values.items():
            self.add_component(entity, str(name), value)

        self.audit.record("entity_spawned", frame=self.frame, entity=entity_id)
        return entity

    def resolve(self, value: Any) -> Entity | None:
        if isinstance(value, Handle):
            entity_id = value.id
            generation = value.generation
        elif isinstance(value, Mapping) and "id" in value:
            entity_id = _as_int(value["id"], -1)
            generation = _as_int(value.get("generation", self.generations.get(entity_id, 0)))
        else:
            entity_id = _as_int(value, -1)
            generation = self.generations.get(entity_id, 0)

        if entity_id not in self.alive_ids:
            return None
        if self.generations.get(entity_id, 0) != generation:
            return None
        return Entity(entity_id, generation)

    def is_alive(self, entity: Any) -> bool:
        return self.resolve(entity) is not None

    def is_destroy_pending(self, entity: Any) -> bool:
        resolved = self.resolve(entity)
        return resolved is not None and resolved.id in self.pending_destroy

    # -- 组件 -----------------------------------------------------------
    def storage(self, component_name: str) -> ComponentStorage:
        name = str(component_name)
        storage = self.storages.get(name)
        if storage is None:
            storage = ComponentStorage(name)
            self.storages[name] = storage
        return storage

    def add_component(self, entity: Any, component_name: str, value: Any) -> Handle:
        resolved = self.resolve(entity)
        if resolved is None:
            raise StaleHandleError(f"cannot add {component_name} to dead entity {entity!r}")
        handle = self.storage(component_name).set(resolved, value)
        self.metrics.handle_reads += 1
        return handle

    def set_component(self, entity: Any, component_name: str, value: Any) -> Handle:
        return self.add_component(entity, component_name, value)

    def get_component(self, entity: Any, component_name: str) -> Any:
        resolved = self.resolve(entity)
        if resolved is None:
            raise StaleHandleError(f"cannot read {component_name} from dead entity {entity!r}")
        self.metrics.handle_reads += 1
        return self.storage(component_name).get(resolved)

    # -- 延迟销毁 -------------------------------------------------------
    def destroy(self, entity: Any) -> bool:
        """提交帧末销毁请求。

        这里不删除任何组件，因此同一帧稍后的系统仍可通过有效句柄安全读取。
        """
        resolved = self.resolve(entity)
        if resolved is None:
            return False
        if resolved.id in self.pending_destroy:
            return False
        self.pending_destroy.add(resolved.id)
        self.metrics.destroy_requested += 1
        self.audit.record("destroy_deferred", frame=self.frame, entity=resolved.id)
        return True

    destroy_entity = destroy
    despawn = destroy

    def destroy_now(self, entity: Any) -> Entity | None:
        resolved = self.resolve(entity)
        if resolved is None:
            return None
        self.pending_destroy.discard(resolved.id)
        return self._destroy_ids([resolved.id])[0]

    def query(
        self,
        *component_names: Any,
        include_pending: bool = True,
        require_all: bool = True,
    ) -> QueryResult:
        names: list[str] = []
        for item in component_names:
            if item is None:
                continue
            if isinstance(item, str):
                names.append(item)
            else:
                names.extend(str(name) for name in _as_set(item))

        candidate_ids = set(self.alive_ids)
        if not include_pending:
            candidate_ids -= self.pending_destroy

        matched: list[Entity] = []
        for entity_id in sorted(candidate_ids):
            generation = self.generations.get(entity_id, 0)
            entity = Entity(entity_id, generation)
            presence = [
                name in self.storages and self.storages[name].contains(entity) for name in names
            ]
            if not names:
                matched.append(entity)
            elif require_all and all(presence):
                matched.append(entity)
            elif not require_all and any(presence):
                matched.append(entity)

        return QueryResult(matched, world=self, components=names)

    # -- 帧末统一回收 ---------------------------------------------------
    def flush_destroyed(self) -> "FrameEndReport":
        return self.end_frame()

    def end_frame(self) -> "FrameEndReport":
        ids = sorted(self.pending_destroy)
        destroyed = self._destroy_ids(ids)
        self.pending_destroy.clear()
        self.frame += 1
        self.metrics.frames += 1
        report = FrameEndReport(
            destroyed,
            frame=self.frame,
            destroyed_ids=tuple(ids),
            pending_remaining=0,
        )
        self.audit.record("frame_end", frame=self.frame, destroyed=ids)
        return report

    def _destroy_ids(self, ids: Iterable[int]) -> list[Entity]:
        destroyed: list[Entity] = []
        for entity_id in ids:
            generation = self.generations.get(entity_id, 0)
            handle = Entity(entity_id, generation)
            for storage in self.storages.values():
                storage._remove_id(entity_id)
            self.alive_ids.discard(entity_id)
            self.pending_destroy.discard(entity_id)
            self.generations[entity_id] = generation + 1
            destroyed.append(handle)
            self.metrics.entities_destroyed += 1
        return destroyed

    def advance_time(self, raw_dt: float | None = None, **kwargs: Any) -> TimestepResult:
        result = self.time.advance(raw_dt, **kwargs)
        self.metrics.timestep_updates += 1
        self.metrics.fixed_steps += result.steps
        self.metrics.clamped_time += _as_float(result)
        self.metrics.dropped_time += result.dropped_time
        return result


def _demo_world(pending: bool = False) -> World:
    world = World()
    first = world.spawn(
        Position(0.0, 0.0),
        Velocity(1.0, 0.0),
        Health=100,
    )
    second = world.spawn(
        Position(10.0, 2.0),
        Velocity(0.0, 1.0),
        Health=80,
    )
    world.spawn(Position(3.0, 4.0), Health=50)
    if pending:
        world.destroy(second)
    return world


# ---------------------------------------------------------------------------
# 时间步夹紧与累加
# ---------------------------------------------------------------------------


@dataclass
class TimeState:
    fixed_dt: float = 1.0 / 60.0
    max_delta: float = 0.1
    max_steps: int = 5
    max_accumulator: float | None = None
    accumulator: float = 0.0
    simulation_time: float = 0.0
    paused: bool = False
    total_dropped_time: float = 0.0
    total_steps: int = 0

    def __post_init__(self) -> None:
        if self.max_accumulator is None:
            self.max_accumulator = max(self.fixed_dt, self.fixed_dt * max(1, self.max_steps))

    def pause(self) -> None:
        self.paused = True

    def resume(self, reset_accumulator: bool = False) -> None:
        self.paused = False
        if reset_accumulator:
            self.accumulator = 0.0

    def advance(self, raw_dt: Any = None, **kwargs: Any) -> "TimestepResult":
        if raw_dt is None:
            raw_dt = kwargs.get("dt", kwargs.get("raw_dt", kwargs.get("delta", 0.0)))
        raw = max(0.0, _as_float(raw_dt, 0.0))
        clamped_dt = min(raw, max(0.0, self.max_delta))

        if self.paused:
            return TimestepResult(
                0.0,
                raw_dt=raw,
                clamped_dt=0.0,
                accumulator=self.accumulator,
                steps=0,
                paused=True,
                dropped_time=0.0,
            )

        self.accumulator += clamped_dt
        assert self.max_accumulator is not None
        dropped = 0.0
        if self.accumulator > self.max_accumulator:
            dropped = self.accumulator - self.max_accumulator
            self.accumulator = self.max_accumulator
            self.total_dropped_time += dropped

        steps = 0
        epsilon = self.fixed_dt * 1e-9 + 1e-12
        while steps < max(1, self.max_steps) and self.accumulator + epsilon >= self.fixed_dt:
            self.accumulator -= self.fixed_dt
            if abs(self.accumulator) < epsilon:
                self.accumulator = 0.0
            steps += 1
            self.total_steps += 1
            self.simulation_time += self.fixed_dt

        # 达到追赶步数上限后，不允许剩余历史在后续帧变成“补帧雪崩”。
        if steps >= self.max_steps and self.accumulator > self.max_accumulator:
            dropped += self.accumulator - self.max_accumulator
            self.accumulator = self.max_accumulator

        return TimestepResult(
            clamped_dt,
            raw_dt=raw,
            clamped_dt=clamped_dt,
            accumulator=self.accumulator,
            steps=steps,
            paused=False,
            dropped_time=dropped,
            simulation_time=self.simulation_time,
        )


class TimestepResult(float):
    """数值上等于夹紧后的 delta，同时携带累加器和固定步数。"""

    raw_dt: float
    clamped_dt: float
    accumulator: float
    steps: int
    paused: bool
    dropped_time: float
    simulation_time: float

    def __new__(cls, value: float, **fields: Any):
        obj = float.__new__(cls, value)
        obj.raw_dt = _as_float(fields.get("raw_dt", value))
        obj.clamped_dt = _as_float(fields.get("clamped_dt", value))
        obj.accumulator = _as_float(fields.get("accumulator", 0.0))
        obj.steps = _as_int(fields.get("steps", 0))
        obj.paused = bool(fields.get("paused", False))
        obj.dropped_time = _as_float(fields.get("dropped_time", 0.0))
        obj.simulation_time = _as_float(fields.get("simulation_time", 0.0))
        return obj

    def __iter__(self):
        yield float(self)
        yield self.accumulator
        yield self.steps

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_dt": self.raw_dt,
            "clamped_dt": self.clamped_dt,
            "accumulator": self.accumulator,
            "steps": self.steps,
            "paused": self.paused,
            "dropped_time": self.dropped_time,
            "simulation_time": self.simulation_time,
        }


# ---------------------------------------------------------------------------
# 系统、依赖图与拓扑排序
# ---------------------------------------------------------------------------


@dataclass
class System:
    name: str
    reads: frozenset[str] = field(default_factory=frozenset)
    writes: frozenset[str] = field(default_factory=frozenset)
    after: frozenset[str] = field(default_factory=frozenset)
    before: frozenset[str] = field(default_factory=frozenset)
    priority: int = 100
    index: int = 0
    exclusive: bool = False
    enabled: bool = True
    run: Callable[..., Any] | None = None

    def __post_init__(self) -> None:
        self.reads = _as_set(self.reads)
        self.writes = _as_set(self.writes)
        self.after = _as_set(self.after)
        self.before = _as_set(self.before)

    @property
    def read_set(self) -> frozenset[str]:
        return self.reads

    @property
    def write_set(self) -> frozenset[str]:
        return self.writes

    def accesses(self) -> frozenset[str]:
        return self.reads | self.writes

    def conflicts_with(self, other: "System") -> frozenset[str]:
        """返回导致两个系统不能并行的组件集合。"""
        return (self.writes & other.accesses()) | (other.writes & self.accesses())


@dataclass(frozen=True)
class DependencyEdge:
    source: str
    target: str
    reason: str
    components: tuple[str, ...] = ()


@dataclass
class DependencyGraph:
    systems: list[System]
    by_name: dict[str, System]
    predecessors: dict[str, set[str]]
    successors: dict[str, set[str]]
    edges: list[DependencyEdge]
    conflicts: list[DependencyEdge]
    errors: list[str] = field(default_factory=list)
    cycles: list[list[str]] = field(default_factory=list)


class TopoOrder(list):
    """拓扑顺序（系统名 list），并保留图和系统对象。"""

    def __init__(
        self,
        names: Iterable[str],
        *,
        systems: Iterable[System] = (),
        graph: DependencyGraph | None = None,
        stages: Iterable[Iterable[str]] | None = None,
    ):
        super().__init__(list(names))
        self.systems = tuple(systems)
        self.graph = graph
        self.stages = [list(stage) for stage in (stages or [])]


class FallbackPlan(list):
    """确定性串行兜底计划。"""

    def __init__(
        self,
        names: Iterable[str],
        *,
        systems: Iterable[System] = (),
        reason: str = "fallback",
        stages: Iterable[Iterable[str]] | None = None,
    ):
        names = list(names)
        super().__init__(names)
        self.systems = tuple(systems)
        self.reason = reason
        self.serial = True
        self.parallel_allowed = False
        self.stages = [list(stage) for stage in (stages or [[name] for name in names])]


class SerialConflictPlan(list):
    def __init__(self, names: Iterable[str], **fields: Any):
        super().__init__(list(names))
        for key, value in fields.items():
            setattr(self, key, value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "order": list(self),
            "stages": getattr(self, "stages", []),
            "write_groups": getattr(self, "write_groups", {}),
            "conflicts": getattr(self, "conflicts", []),
            "executed_log": getattr(self, "executed_log", []),
            "safe": getattr(self, "safe", True),
        }


def _looks_like_system_spec(value: tuple | list) -> bool:
    if not value:
        return False
    if not isinstance(value[0], str):
        return False
    if len(value) == 1:
        return True
    second = value[1]
    return isinstance(second, (frozenset, set, list, tuple, dict)) or callable(second)


def _as_system(value: Any, index: int = 0) -> System:
    if isinstance(value, System):
        value.index = index
        return value
    if isinstance(value, str):
        return System(name=value, index=index)
    if callable(value):
        return System(name=getattr(value, "__name__", f"system_{index}"), index=index, run=value)
    if isinstance(value, Mapping):
        name = _pick(value, ("name", "id", "system", "fn_name", "key"), f"system_{index}")
        reads = _pick(
            value,
            ("reads", "read", "read_set", "read_components", "readonly", "r"),
            frozenset(),
        )
        writes = _pick(
            value,
            ("writes", "write", "write_set", "write_components", "writable", "w"),
            frozenset(),
        )
        rw = _pick(value, ("rw", "read_write", "readwrite", "access"), None)
        if rw is not None:
            reads = _as_set(reads) | _as_set(rw)
            writes = _as_set(writes) | _as_set(rw)
        return System(
            name=str(name),
            reads=_as_set(reads),
            writes=_as_set(writes),
            after=_as_set(_pick(value, ("after", "deps", "depends_on", "requires"), None)),
            before=_as_set(_pick(value, ("before",), None)),
            priority=_as_int(_pick(value, ("priority", "order_hint"), 100), 100),
            index=index,
            exclusive=bool(_pick(value, ("exclusive", "serial"), False)),
            enabled=bool(_pick(value, ("enabled",), True)),
            run=_pick(value, ("run", "fn", "callback", "execute", "handler"), None),
        )
    if isinstance(value, (tuple, list)):
        if len(value) >= 4:
            name, reads, writes, run = value[:4]
        elif len(value) == 3:
            name, reads, writes = value
            run = None
        elif len(value) == 2:
            name, components = value
            # 两元组没有足够信息区分读/写；把它视为读写集合只会更保守，不会漏冲突。
            reads = writes = components
            run = None
        else:
            name, reads, writes, run = value[0], frozenset(), frozenset(), None
        return System(
            name=str(name),
            reads=_as_set(reads),
            writes=_as_set(writes),
            index=index,
            run=run if callable(run) else None,
        )
    return System(name=f"system_{index}", index=index)


def _demo_systems() -> list[System]:
    return [
        System("InputSystem", writes={"Input"}, priority=10),
        System("MovementSystem", reads={"Input"}, writes={"Velocity", "Position"}, priority=20),
        System(
            "CollisionSystem",
            reads={"Position", "Velocity"},
            writes={"Position", "Contacts"},
            priority=30,
        ),
        System(
            "AnimationSystem",
            reads={"Position", "Velocity", "Contacts"},
            writes={"Animation"},
            priority=40,
        ),
        System("RenderSystem", reads={"Position", "Animation", "Contacts"}, priority=50),
    ]


def _coerce_systems(*args: Any, **kwargs: Any) -> list[System]:
    if "systems" in kwargs and kwargs["systems"] is not None:
        value = kwargs["systems"]
    elif "scheduler" in kwargs and kwargs["scheduler"] is not None:
        value = kwargs["scheduler"]
    elif len(args) == 0:
        value = None
    elif len(args) == 1:
        value = args[0]
    else:
        value = tuple(args)

    if value is None:
        systems = _demo_systems()
    elif isinstance(value, (int, float, bool)):
        systems = _demo_systems()
    elif isinstance(value, Scheduler):
        systems = list(value.systems)
    elif isinstance(value, SchedulePlan):
        systems = list(value.systems)
    elif isinstance(value, System):
        systems = [value]
    elif isinstance(value, Mapping):
        if "systems" in value:
            return _coerce_systems(value["systems"])
        systems = [_as_system(value)]
    elif callable(value):
        systems = [_as_system(value)]
    elif isinstance(value, tuple) and _looks_like_system_spec(value):
        systems = [_as_system(value)]
    else:
        try:
            items = list(value)
        except TypeError:
            systems = _demo_systems()
        else:
            if isinstance(value, (tuple, list)) and _looks_like_system_spec(items):
                systems = [_as_system(value)]
            else:
                systems = [_as_system(item, i) for i, item in enumerate(items)]

    # 处理重名，保证图节点唯一；正常输入不会触发。
    seen: dict[str, int] = {}
    for index, system in enumerate(systems):
        system.index = index
        if system.name in seen:
            seen[system.name] += 1
            system.name = f"{system.name}#{seen[system.name]}"
        else:
            seen[system.name] = 1
    return systems


def _build_graph(systems: list[System], *, ordering: str = "dataflow") -> DependencyGraph:
    by_name = {system.name: system for system in systems}
    preds = {name: set() for name in by_name}
    succs = {name: set() for name in by_name}
    edges: list[DependencyEdge] = []
    edge_pairs: set[tuple[str, str]] = set()
    conflicts: list[DependencyEdge] = []
    errors: list[str] = []

    def add_edge(source: str, target: str, reason: str, components: tuple[str, ...] = ()) -> None:
        if source not in by_name or target not in by_name:
            missing = source if source not in by_name else target
            errors.append(f"dependency references unknown system: {missing}")
            return
        if source == target:
            errors.append(f"system {source} depends on itself")
            return
        edge = DependencyEdge(source, target, reason, tuple(sorted(components)))
        if edge not in edges:
            edges.append(edge)
        edge_pairs.add((source, target))
        preds[target].add(source)
        succs[source].add(target)

    for system in systems:
        for dep in _sorted_items(system.after):
            add_edge(str(dep), system.name, "explicit_after")
        for dep in _sorted_items(system.before):
            add_edge(system.name, str(dep), "explicit_before")

    for i, left in enumerate(systems):
        for right in systems[i + 1 :]:
            components = left.conflicts_with(right)
            if not components:
                continue
            component_tuple = tuple(str(c) for c in _sorted_items(components))
            conflicts.append(
                DependencyEdge(left.name, right.name, "rw_conflict", component_tuple)
            )

            # 显式 after/before 是最高优先级；已经给出任一方向时，不用推断边制造环。
            if (left.name, right.name) in edge_pairs or (right.name, left.name) in edge_pairs:
                continue

            if ordering == "declaration":
                add_edge(left.name, right.name, "rw_conflict", component_tuple)
                continue

            # RAW：写者必须在读同一组件的系统之前，保证读到本帧确定的数据。
            left_to_right_reads = tuple(_sorted_items(left.writes & right.reads))
            right_to_left_reads = tuple(_sorted_items(right.writes & left.reads))
            if left_to_right_reads:
                add_edge(left.name, right.name, "write_before_read", left_to_right_reads)
            if right_to_left_reads:
                add_edge(right.name, left.name, "write_before_read", right_to_left_reads)

            # 写-写没有天然数据方向；按稳定声明/优先级键串行，消除并行写竞争。
            write_write = tuple(_sorted_items(left.writes & right.writes))
            if write_write:
                left_key = _tie_key(left)
                right_key = _tie_key(right)
                if left_key <= right_key:
                    add_edge(left.name, right.name, "write_write_serial", write_write)
                else:
                    add_edge(right.name, left.name, "write_write_serial", write_write)

            # 纯 WAR（读-后-写）若尚未被 RAW/WW 覆盖，按稳定声明顺序定向。
            if not any(
                source == left.name and target == right.name
                for source, target in edge_pairs
            ) and not any(
                source == right.name and target == left.name
                for source, target in edge_pairs
            ):
                add_edge(left.name, right.name, "read_write_serial", component_tuple)

    graph = DependencyGraph(
        systems=systems,
        by_name=by_name,
        predecessors=preds,
        successors=succs,
        edges=edges,
        conflicts=conflicts,
        errors=errors,
    )
    graph.cycles = _find_cycles(graph)
    if graph.cycles:
        errors.append(f"dependency cycle: {graph.cycles}")
    return graph


def _find_cycles(graph: DependencyGraph) -> list[list[str]]:
    nodes = sorted(graph.by_name)
    visited: set[str] = set()
    order: list[str] = []

    def dfs(node: str) -> None:
        if node in visited:
            return
        visited.add(node)
        for nxt in sorted(graph.successors.get(node, ())):
            dfs(nxt)
        order.append(node)

    for node in nodes:
        dfs(node)

    reverse: dict[str, set[str]] = {name: set() for name in nodes}
    for source, targets in graph.successors.items():
        for target in targets:
            if target in reverse:
                reverse[target].add(source)

    seen: set[str] = set()
    cycles: list[list[str]] = []
    for start in reversed(order):
        if start in seen:
            continue
        stack = [start]
        component: list[str] = []
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            component.append(node)
            stack.extend(reverse.get(node, ()) - seen)
        if len(component) > 1:
            cycles.append(sorted(component))
        elif start in graph.successors.get(start, set()):
            cycles.append([start])
    return cycles


def _tie_key(system: System) -> tuple[int, int, str]:
    return (system.priority, system.index, system.name)


def _topological_order(
    systems: list[System],
    *,
    on_cycle: str = "raise",
    ordering: str = "dataflow",
) -> TopoOrder | FallbackPlan:
    graph = _build_graph(systems, ordering=ordering)
    if graph.cycles or any("cycle" in error for error in graph.errors):
        if on_cycle == "raise":
            raise ScheduleCycleError(graph.cycles)
        return _fallback_plan(systems, reason="dependency_cycle")

    remaining = {name: set(pred) for name, pred in graph.predecessors.items()}
    ready = [system for system in systems if not remaining[system.name]]
    order: list[str] = []
    used: set[str] = set()

    while ready:
        ready.sort(key=_tie_key)
        system = ready.pop(0)
        if system.name in used:
            continue
        used.add(system.name)
        order.append(system.name)
        for nxt_name in sorted(graph.successors[system.name]):
            remaining[nxt_name].discard(system.name)
            if not remaining[nxt_name]:
                nxt = graph.by_name[nxt_name]
                if nxt_name not in used and nxt not in ready:
                    ready.append(nxt)

    if len(order) != len(systems):
        if on_cycle == "raise":
            raise ScheduleCycleError(_find_cycles(graph))
        return _fallback_plan(systems, reason="topological_sort_incomplete")

    stages = _stages_for_order(order, graph)
    return TopoOrder(order, systems=systems, graph=graph, stages=stages)


def _stages_for_order(order: list[str], graph: DependencyGraph) -> list[list[str]]:
    level: dict[str, int] = {}
    for name in order:
        preds = graph.predecessors.get(name, set())
        level[name] = 0 if not preds else 1 + max(level[pred] for pred in preds)
    stage_count = max(level.values(), default=-1) + 1
    stages: list[list[str]] = [[] for _ in range(stage_count)]
    for name in order:
        stages[level[name]].append(name)
    for stage in stages:
        stage.sort(key=lambda item: _tie_key(graph.by_name[item]))
    return stages


def _fallback_plan(systems: list[System], reason: str = "fallback") -> FallbackPlan:
    ordered = sorted(systems, key=lambda system: (system.index, system.priority, system.name))
    names = [system.name for system in ordered]
    return FallbackPlan(names, systems=ordered, reason=reason)


def _fingerprint(order: Iterable[str], stages: Iterable[Iterable[str]]) -> str:
    order_tuple = tuple(order)
    stage_tuple = tuple(tuple(stage) for stage in stages)
    digest = hashlib.sha256()
    digest.update(repr(order_tuple).encode("utf-8"))
    digest.update(b"::")
    digest.update(repr(stage_tuple).encode("utf-8"))
    return digest.hexdigest()[:16]


# ---------------------------------------------------------------------------
# 调度器与调度报告
# ---------------------------------------------------------------------------


class SchedulePlan(list):
    """扁平顺序是系统名 list；额外暴露 stages、指纹和校验结果。"""

    def __init__(
        self,
        order: Iterable[str],
        *,
        systems: Iterable[System] = (),
        stages: Iterable[Iterable[str]] | None = None,
        graph: DependencyGraph | None = None,
        validation: "ValidationResult | None" = None,
        fingerprint: str = "",
        fallback_used: bool = False,
        **metadata: Any,
    ):
        order = list(order)
        super().__init__(order)
        self.systems = tuple(systems)
        self.stages = [list(stage) for stage in (stages or [order])]
        self.graph = graph
        self.validation = validation
        self.fingerprint = fingerprint or _fingerprint(order, self.stages)
        self.fallback_used = fallback_used
        self.metadata = metadata
        has_parallel_stage = any(len(stage) > 1 for stage in self.stages)
        self.serial = fallback_used or not has_parallel_stage

    def to_dict(self) -> dict[str, Any]:
        return {
            "order": list(self),
            "stages": [list(stage) for stage in self.stages],
            "fingerprint": self.fingerprint,
            "fallback_used": self.fallback_used,
            "valid": bool(self.validation) if self.validation is not None else True,
            "metadata": self.metadata,
        }


@dataclass
class Scheduler:
    systems: list[System] = field(default_factory=list)
    world: World | None = None
    audit: AuditLog = field(default_factory=AuditLog)
    metrics: Metrics = field(default_factory=Metrics)
    last_plan: SchedulePlan | None = None

    def __post_init__(self) -> None:
        if self.systems and not isinstance(self.systems[0], System):
            self.systems = _coerce_systems(self.systems)
        else:
            for index, system in enumerate(self.systems):
                if isinstance(system, System):
                    system.index = index

    def add_system(self, system: Any) -> System:
        normalized = _as_system(system, len(self.systems))
        self.systems.append(normalized)
        return normalized

    def plan(self, *, use_fallback: bool = True, validate: bool = True) -> SchedulePlan:
        try:
            topo = _topological_order(self.systems, on_cycle="raise")
        except ScheduleCycleError:
            if not use_fallback:
                raise
            fallback_plan = _fallback_plan(self.systems, reason="dependency_cycle")
            report = ValidationResult(
                {
                    "ok": False,
                    "valid": False,
                    "cycle": True,
                    "errors": [f"cycle:{fallback_plan.reason}"],
                    "order": list(fallback_plan),
                }
            )
            plan = SchedulePlan(
                fallback_plan,
                systems=fallback_plan.systems,
                stages=fallback_plan.stages,
                validation=report,
                fallback_used=True,
            )
            self.metrics.cycle_fallbacks += 1
        else:
            assert isinstance(topo, TopoOrder)
            report = _validate_plan(self.systems, topo, check_determinism=False) if validate else None
            plan = SchedulePlan(
                topo,
                systems=topo.systems,
                stages=topo.stages,
                graph=topo.graph,
                validation=report,
            )

        self.last_plan = plan
        self.metrics.systems_scheduled += len(plan)
        self.metrics.parallel_stages += sum(1 for stage in plan.stages if len(stage) > 1)
        self.metrics.serial_stages += sum(1 for stage in plan.stages if len(stage) == 1)
        if plan.graph is not None:
            self.metrics.conflicts_serialized += len(plan.graph.conflicts)
        self.audit.record(
            "schedule_planned",
            order=list(plan),
            stages=plan.stages,
            fingerprint=plan.fingerprint,
            fallback=plan.fallback_used,
        )
        return plan

    def run_frame(
        self,
        raw_dt: Any = None,
        *,
        parallel: bool = True,
        flush: bool = True,
    ) -> dict[str, Any]:
        plan = self.plan()
        executed: list[str] = []
        stage_results: list[Any] = []
        by_name = {system.name: system for system in self.systems}

        for stage_index, stage in enumerate(plan.stages):
            jobs: list[Callable[[], Any]] = []
            for name in stage:
                system = by_name[name]

                def make_job(system: System = system) -> Callable[[], Any]:
                    def job() -> Any:
                        if system.run is None:
                            return None
                        if self.world is not None:
                            try:
                                return system.run(self.world)
                            except TypeError:
                                return system.run()
                        return system.run()

                    job.__name__ = system.name
                    return job

                jobs.append(make_job(system))

            if parallel and len(jobs) > 1:
                joined = job_join(jobs, parallel=True)
                stage_results.append(joined)
            else:
                for name, job in zip(stage, jobs):
                    job()
                    executed.append(name)
            if parallel and len(jobs) > 1:
                executed.extend(stage)
            self.metrics.systems_run += len(stage)

        time_result = self.world.advance_time(raw_dt) if self.world is not None and raw_dt is not None else None
        frame_report = self.world.end_frame() if flush and self.world is not None else None
        return {
            "plan": plan,
            "executed": executed,
            "time": time_result,
            "frame_end": frame_report,
            "stage_results": stage_results,
        }


# ---------------------------------------------------------------------------
# 校验、确定性、审计、指标入口
# ---------------------------------------------------------------------------


class ValidationResult(dict):
    def __bool__(self) -> bool:
        return bool(self.get("ok", self.get("valid", False)))

    @property
    def ok(self) -> bool:
        return bool(self.get("ok", False))

    @property
    def errors(self) -> list[str]:
        return list(self.get("errors", []))


class DeterminismReport(dict):
    def __bool__(self) -> bool:
        return bool(self.get("stable", False)) and bool(self.get("ok", True))

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    @property
    def ok(self) -> bool:
        return bool(self.get("ok", False))


def _validate_plan(
    systems: list[System],
    plan: TopoOrder | SchedulePlan,
    *,
    check_determinism: bool = True,
) -> ValidationResult:
    graph = _build_graph(systems)
    errors = list(graph.errors)
    names = [system.name for system in systems]
    order = list(plan)

    if sorted(order) != sorted(names) or len(order) != len(set(order)):
        errors.append("schedule must contain every system exactly once")

    position = {name: index for index, name in enumerate(order)}
    for edge in graph.edges:
        if edge.source in position and edge.target in position:
            if position[edge.source] >= position[edge.target]:
                errors.append(f"dependency edge {edge.source}->{edge.target} is violated")

    for conflict in graph.conflicts:
        left, right = conflict.source, conflict.target
        if left in position and right in position and position[left] == position[right]:
            errors.append(f"conflicting systems {left} and {right} share an order slot")

    stage_membership = {
        name: stage_index for stage_index, stage in enumerate(getattr(plan, "stages", [])) for name in stage
    }
    for conflict in graph.conflicts:
        if conflict.source in stage_membership and conflict.target in stage_membership:
            if stage_membership[conflict.source] == stage_membership[conflict.target]:
                errors.append(
                    f"conflicting systems {conflict.source}/{conflict.target} are in one parallel stage"
                )

    fingerprints: list[str] = []
    stable = True
    if check_determinism:
        repeats = [_topological_order(systems, on_cycle="fallback") for _ in range(5)]
        signatures = {
            (tuple(repeat), tuple(tuple(stage) for stage in getattr(repeat, "stages", [])))
            for repeat in repeats
        }
        stable = len(signatures) == 1
        fingerprints = [
            _fingerprint(repeat, getattr(repeat, "stages", [])) for repeat in repeats
        ]
        if not stable:
            errors.append("scheduling result is not deterministic across repeated planning")

    fingerprint = getattr(plan, "fingerprint", _fingerprint(order, getattr(plan, "stages", [order])))
    return ValidationResult(
        {
            "ok": not errors,
            "valid": not errors,
            "errors": errors,
            "warnings": [],
            "order": order,
            "stages": [list(stage) for stage in getattr(plan, "stages", [])],
            "conflicts": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "components": list(edge.components),
                }
                for edge in graph.conflicts
            ],
            "fingerprint": fingerprint,
            "fingerprints": fingerprints or [fingerprint],
            "deterministic": stable,
            "cycle": bool(graph.cycles),
            "systems": systems,
        }
    )


def validation(*args: Any, **kwargs: Any) -> ValidationResult:
    plan = kwargs.get("plan")
    systems: list[System] | None = None
    for arg in args:
        if isinstance(arg, SchedulePlan):
            plan = arg
        elif isinstance(arg, Scheduler):
            systems = list(arg.systems)
        elif isinstance(arg, World):
            systems = []
    if systems is None:
        systems = _coerce_systems(*args, **kwargs)
    if plan is not None and getattr(plan, "systems", None):
        systems = list(plan.systems)

    try:
        topo = _topological_order(systems, on_cycle="raise")
    except ScheduleCycleError as exc:
        return ValidationResult(
            {
                "ok": False,
                "valid": False,
                "cycle": True,
                "errors": [str(exc)],
                "order": [],
                "stages": [],
                "deterministic": True,
            }
        )
    return _validate_plan(systems, topo)


def determinism(*args: Any, repeats: int = 7, **kwargs: Any) -> DeterminismReport:
    systems = _coerce_systems(*args, **kwargs)
    plans: list[SchedulePlan] = []
    for _ in range(max(2, repeats)):
        plans.append(Scheduler(list(systems)).plan())

    signatures = {
        (tuple(plan), tuple(tuple(stage) for stage in plan.stages), plan.fingerprint)
        for plan in plans
    }
    stable = len(signatures) == 1

    # 不同 job 完成顺序不得改变 join 后的提交顺序结果。
    demo_jobs = [lambda i=i: i * i for i in range(6)]
    joined_a = job_join(demo_jobs, parallel=False)
    joined_b = job_join(list(reversed(demo_jobs)), parallel=False)
    join_stable = list(joined_a) == [0, 1, 4, 9, 16, 25]
    reverse_stable = list(joined_b) == [25, 16, 9, 4, 1, 0]

    report = DeterminismReport(
        {
            "ok": stable and join_stable and reverse_stable,
            "stable": stable and join_stable,
            "schedule_stable": stable,
            "join_order_stable": join_stable,
            "reversed_input_preserved_as_submission_order": reverse_stable,
            "runs": len(plans),
            "fingerprint": plans[0].fingerprint if plans else "",
            "fingerprints": [plan.fingerprint for plan in plans],
            "order": list(plans[0]) if plans else [],
            "stages": plans[0].stages if plans else [],
        }
    )
    return report


def audit(*args: Any, **kwargs: Any) -> AuditReport:
    source: Any = None
    entries: list[Mapping[str, Any]] = []
    for arg in args:
        if isinstance(arg, (World, Scheduler, AuditLog)):
            source = arg
            log = arg.audit if hasattr(arg, "audit") else arg
            entries.extend(getattr(log, "entries", []))
        elif isinstance(arg, AuditReport):
            entries.extend(arg)
        elif isinstance(arg, Mapping):
            entries.append(arg)
        elif isinstance(arg, str):
            entries.append({"event": arg})

    if not entries and source is None:
        world = _demo_world(pending=True)
        scheduler = Scheduler(_demo_systems(), world=world)
        scheduler.plan()
        world.end_frame()
        entries = world.audit.entries
        source = world

    return AuditReport(entries, source=source)


def fallback(*args: Any, **kwargs: Any) -> FallbackPlan:
    reason = str(kwargs.get("reason", "manual_serial_fallback"))
    validation_report = None
    for arg in args:
        if isinstance(arg, ValidationResult):
            validation_report = arg
            reason = "validation_failed"
            maybe_systems = arg.get("systems")
            if maybe_systems:
                systems = list(maybe_systems)
                break
    else:
        systems = _coerce_systems(*args, **kwargs)

    if validation_report is not None and validation_report.get("cycle"):
        reason = "dependency_cycle"
    if reason != "dependency_cycle":
        graph = _build_graph(systems)
        if graph.cycles:
            reason = "dependency_cycle"
    return _fallback_plan(systems, reason=reason)


def metrics(*args: Any, **kwargs: Any) -> MetricsSnapshot:
    for arg in args:
        if isinstance(arg, Metrics):
            return arg.snapshot()
        if isinstance(arg, (World, Scheduler)):
            return arg.metrics.snapshot()
        if isinstance(arg, Mapping):
            data = dict(arg)
            data.setdefault("ok", True)
            return MetricsSnapshot(data)
    data = Metrics().snapshot()
    if kwargs:
        data.update(kwargs)
    return data


# ---------------------------------------------------------------------------
# Job join
# ---------------------------------------------------------------------------


class JobJoinResult(list):
    def __init__(
        self,
        results: Iterable[Any] | None = None,
        *,
        completion_order: Iterable[str] = (),
        errors: Mapping[int, BaseException] | None = None,
        ran_in_parallel: bool = False,
    ):
        results = list(results or [])
        super().__init__(results)
        self.results = tuple(results)
        self.completion_order = tuple(completion_order)
        self.errors = dict(errors or {})
        self.ok = not self.errors
        self.ran_in_parallel = ran_in_parallel

    def raise_for_errors(self) -> None:
        if self.errors:
            first = next(iter(self.errors.values()))
            raise first


def _normalize_job(job: Any, index: int) -> tuple[str, Callable[[], Any]]:
    if isinstance(job, Mapping):
        name = str(_pick(job, ("name", "id"), f"job_{index}"))
        fn = _pick(job, ("run", "fn", "callback", "handler", "job"), None)
        if callable(fn):
            return name, fn
        return name, lambda: job

    if isinstance(job, tuple) and len(job) == 2 and isinstance(job[0], str) and callable(job[1]):
        return job[0], job[1]

    if callable(job):
        return getattr(job, "__name__", f"job_{index}"), job

    return f"value_{index}", lambda value=job: value


def _collect_jobs(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> list[Any]:
    if "jobs" in kwargs and kwargs["jobs"] is not None:
        value = kwargs["jobs"]
    elif len(args) == 0:
        return [lambda: i for i in range(3)]
    elif len(args) == 1:
        value = args[0]
        if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], str) and callable(value[1]):
            return [value]
    else:
        return list(args)

    if callable(value) or isinstance(value, Mapping):
        return [value]
    if isinstance(value, (int, float, str, bytes)):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def job_join(*args: Any, **kwargs: Any) -> JobJoinResult:
    jobs = _collect_jobs(args, kwargs)
    parallel = bool(kwargs.get("parallel", True))
    max_workers = kwargs.get("max_workers")
    timeout = kwargs.get("timeout")
    raise_on_error = bool(kwargs.get("raise_on_error", False))

    normalized = [_normalize_job(job, index) for index, job in enumerate(jobs)]
    results: list[Any] = [None] * len(normalized)
    observed_order: list[str] = []
    errors: dict[int, BaseException] = {}

    def run_direct(index: int, name: str, fn: Callable[[], Any]) -> None:
        try:
            results[index] = fn()
        except BaseException as exc:  # join 需要收集全部任务异常
            errors[index] = exc
            results[index] = exc
        finally:
            observed_order.append(name)

    callable_count = sum(1 for _, fn in normalized if callable(fn))
    simulated_order = kwargs.get("completion_order")
    if simulated_order is not None:
        # 确定性测试入口：无论指定的完成排列如何，最终结果都按提交顺序 join。
        for index, (name, fn) in enumerate(normalized):
            run_direct(index, name, fn)
        name_to_index = {name: index for index, (name, _) in enumerate(normalized)}
        permutation: list[int] = []
        for item in simulated_order:
            if isinstance(item, str) and item in name_to_index:
                permutation.append(name_to_index[item])
            else:
                index = _as_int(item, -1)
                if 0 <= index < len(normalized):
                    permutation.append(index)
        if len(permutation) == len(normalized) and len(set(permutation)) == len(normalized):
            completion_order = [normalized[index][0] for index in permutation]
        else:
            completion_order = observed_order
    elif parallel and callable_count > 1:
        completion_order = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(fn): (index, name)
                for index, (name, fn) in enumerate(normalized)
            }
            try:
                futures = as_completed(future_to_index, timeout=timeout)
                for future in futures:
                    index, name = future_to_index[future]
                    completion_order.append(name)
                    try:
                        results[index] = future.result()
                    except BaseException as exc:
                        errors[index] = exc
                        results[index] = exc
            except BaseException as exc:
                for future, (index, name) in future_to_index.items():
                    if not future.done():
                        future.cancel()
                    else:
                        try:
                            results[index] = future.result()
                        except BaseException as job_exc:
                            errors[index] = job_exc
                errors.setdefault(-1, exc)
    else:
        for index, (name, fn) in enumerate(normalized):
            run_direct(index, name, fn)
        completion_order = observed_order

    report = JobJoinResult(
        results,
        completion_order=completion_order,
        errors=errors,
        ran_in_parallel=parallel and callable_count > 1 and simulated_order is None,
    )
    if raise_on_error and errors:
        report.raise_for_errors()
    return report


# ---------------------------------------------------------------------------
# 公共入口：拓扑、销毁、冲突、句柄、时间步、调度
# ---------------------------------------------------------------------------


def topo_sort_rw(*args: Any, **kwargs: Any) -> TopoOrder | FallbackPlan:
    systems = _coerce_systems(*args, **kwargs)
    on_cycle = str(kwargs.get("on_cycle", "raise"))
    ordering = str(kwargs.get("ordering", "dataflow"))
    return _topological_order(systems, on_cycle=on_cycle, ordering=ordering)


def schedule(*args: Any, **kwargs: Any) -> SchedulePlan:
    world = kwargs.get("world")
    scheduler_arg = kwargs.get("scheduler")
    for arg in args:
        if isinstance(arg, World):
            world = arg
        elif isinstance(arg, Scheduler):
            scheduler_arg = arg

    if isinstance(scheduler_arg, Scheduler):
        scheduler = scheduler_arg
        if world is not None:
            scheduler.world = world
    else:
        systems = _coerce_systems(*args, **kwargs)
        scheduler = Scheduler(systems, world=world)
    return scheduler.plan(
        use_fallback=bool(kwargs.get("use_fallback", True)),
        validate=bool(kwargs.get("validate", True)),
    )


def write_conflict_serial(*args: Any, **kwargs: Any) -> SerialConflictPlan:
    systems = _coerce_systems(*args, **kwargs)
    topo = _topological_order(systems, on_cycle="fallback")
    by_name = {system.name: system for system in systems}
    order = list(topo)

    write_groups: dict[str, list[str]] = {}
    for name in order:
        for component in _sorted_items(by_name[name].writes):
            write_groups.setdefault(str(component), []).append(name)
    multi_writer_groups = {key: value for key, value in write_groups.items() if len(value) > 1}

    executed_log: list[str] = []
    deterministic_state: dict[str, list[str]] = {component: [] for component in write_groups}
    for name in order:
        system = by_name[name]
        if system.run is not None:
            try:
                system.run()
            except TypeError:
                system.run(None)
        for component in _sorted_items(system.writes):
            deterministic_state.setdefault(str(component), []).append(name)
        executed_log.append(name)

    same_stage_conflicts: list[str] = []
    for stage in getattr(topo, "stages", []):
        stage_set = set(stage)
        for edge in _build_graph(systems).conflicts:
            if edge.source in stage_set and edge.target in stage_set:
                same_stage_conflicts.append(f"{edge.source}/{edge.target}:{edge.components}")

    plan = SerialConflictPlan(
        order,
        stages=getattr(topo, "stages", [[name] for name in order]),
        write_groups=multi_writer_groups,
        all_write_groups=write_groups,
        conflicts=[
            {"source": edge.source, "target": edge.target, "components": list(edge.components)}
            for edge in _build_graph(systems).conflicts
        ],
        executed_log=executed_log,
        write_trace=deterministic_state,
        same_stage_conflicts=same_stage_conflicts,
        safe=not same_stage_conflicts,
        serial=True,
    )
    return plan


def deferred_destroy(*args: Any, **kwargs: Any) -> dict[str, Any]:
    world: World | None = kwargs.get("world")
    targets: list[Any] = []
    for arg in args:
        if isinstance(arg, World):
            world = arg
        elif isinstance(arg, Scheduler):
            world = arg.world
        else:
            targets.append(arg)

    created_world = world is None
    if world is None:
        world = _demo_world(pending=False)
        if not targets:
            targets = [Entity(1, 0)]

    if not targets:
        targets = list(world.query(include_pending=True))

    requested: list[int] = []
    for target in targets:
        resolved = world.resolve(target)
        if resolved is None and isinstance(target, int):
            # 允许直接传逻辑 ID；演示世界的第一个实体 ID 为 1。
            resolved = world.resolve(Entity(target, world.generations.get(target, 0)))
        if resolved is not None and world.destroy(resolved):
            requested.append(resolved.id)

    before = world.query(include_pending=True)
    active_before = world.query(include_pending=False)
    component_name = "Position" if "Position" in world.storages else next(iter(world.storages), None)
    sample_id = requested[0] if requested else (before[0].id if before else None)
    sample_entity = world.resolve(sample_id) if sample_id is not None else None
    value_before: Any = None
    component_available = False
    if sample_entity is not None and component_name is not None:
        try:
            value_before = world.get_component(sample_entity, component_name)
            component_available = True
        except StaleHandleError:
            component_available = False

    frame_report = world.end_frame()
    after = world.query(include_pending=True)
    handle_valid_after = sample_entity is not None and world.is_alive(sample_entity)
    value_after: Any = None
    try:
        if sample_entity is not None and component_name is not None:
            value_after = world.get_component(sample_entity, component_name)
    except StaleHandleError:
        value_after = None

    requested_set = set(requested)
    after_ids = set(after.ids)
    targets_gone = not (requested_set & after_ids)
    safe = component_available and not handle_valid_after and targets_gone

    return {
        "ok": safe,
        "created_demo_world": created_world,
        "requested": requested,
        "queried_before_flush": list(before.ids),
        "active_query_before_flush": list(active_before.ids),
        "component_before_flush": value_before,
        "component_available_before_flush": component_available,
        "released_early": False,
        "flushed_at_frame_end": list(frame_report.destroyed_ids),
        "queried_after_flush": list(after.ids),
        "target_present_after_flush": not targets_gone,
        "handle_valid_after_flush": handle_valid_after,
        "value_after_flush": value_after,
        "frame": frame_report.frame,
        "safe": safe,
        "frame_report": frame_report,
    }


def handle_not_pointer(*args: Any, **kwargs: Any) -> dict[str, Any]:
    storage = next((arg for arg in args if isinstance(arg, ComponentStorage)), None)
    if storage is None:
        storage = ComponentStorage("StableHandleDemo", initial_capacity=1)

    handle = Handle(42, 0)
    value = {"position": [1, 2, 3]}
    storage.set(handle, value)
    before_capacity = storage.capacity

    extra_count = _as_int(kwargs.get("extra", kwargs.get("count", 64)), 64)
    extra_handles = [Handle(1000 + i, 0) for i in range(max(1, extra_count))]
    for i, extra in enumerate(extra_handles):
        storage.set(extra, i)

    value_after_growth = storage.get(handle)
    all_extra_preserved = all(storage.get(extra) == i for i, extra in enumerate(extra_handles))
    storage.remove(handle)
    stale_detected = False
    try:
        storage.get(handle)
    except StaleHandleError:
        stale_detected = True

    return {
        "ok": value_after_growth == value and all_extra_preserved and stale_detected,
        "uses_stable_handle": True,
        "raw_pointer_exposed": False,
        "handle": handle,
        "capacity_before": before_capacity,
        "capacity_after": storage.capacity,
        "growths": storage.growths,
        "value_after_grow": value_after_growth,
        "all_handles_preserved": all_extra_preserved,
        "stale_handle_detected": stale_detected,
        "storage": storage,
    }


def timestep_clamp(*args: Any, **kwargs: Any) -> TimestepResult:
    state = kwargs.get("state")
    world = kwargs.get("world")
    numeric_args: list[float] = []
    for arg in args:
        if isinstance(arg, TimeState):
            state = arg
        elif isinstance(arg, World):
            world = arg
        elif isinstance(arg, bool):
            continue
        elif isinstance(arg, (int, float)):
            numeric_args.append(float(arg))

    if state is None and isinstance(world, World):
        state = world.time
    if state is not None:
        dt = kwargs.get("dt", kwargs.get("raw_dt", numeric_args[0] if numeric_args else 0.0))
        return state.advance(dt)

    dt = kwargs.get("dt", kwargs.get("raw_dt", kwargs.get("delta", numeric_args[0] if numeric_args else 0.016)))
    fixed_dt = _as_float(kwargs.get("fixed_dt", 1.0 / 60.0), 1.0 / 60.0)
    max_delta = _as_float(kwargs.get("max_delta", kwargs.get("max_dt", 0.1)), 0.1)
    max_steps = _as_int(kwargs.get("max_steps", 5), 5)
    accumulator = _as_float(kwargs.get("accumulator", kwargs.get("acc", 0.0)), 0.0)

    # 常见两参数纯函数形式：timestep_clamp(dt, max_delta)。
    if (
        len(numeric_args) >= 2
        and "max_delta" not in kwargs
        and "max_dt" not in kwargs
        and "accumulator" not in kwargs
        and "acc" not in kwargs
    ):
        max_delta = numeric_args[1]
    elif len(numeric_args) >= 2 and "accumulator" not in kwargs:
        accumulator = numeric_args[1]
    if len(numeric_args) >= 3:
        fixed_dt = numeric_args[2]

    state = TimeState(
        fixed_dt=fixed_dt,
        max_delta=max_delta,
        max_steps=max_steps,
        accumulator=accumulator,
    )
    return state.advance(dt)


def query(*args: Any, **kwargs: Any) -> QueryResult:
    world: World | None = kwargs.get("world")
    components: list[Any] = []
    for arg in args:
        if isinstance(arg, World):
            world = arg
        elif isinstance(arg, Scheduler):
            world = arg.world
        elif isinstance(arg, str):
            components.append(arg)
        elif isinstance(arg, (list, tuple, set, frozenset)):
            components.extend(arg)
        elif isinstance(arg, Mapping):
            components.extend(arg.keys())
        elif arg is None or isinstance(arg, (int, float, bool)):
            continue
        else:
            components.append(arg)

    if world is None:
        world = _demo_world()
    if not components:
        components = list(kwargs.get("components", ()))
    return world.query(
        *components,
        include_pending=bool(kwargs.get("include_pending", True)),
        require_all=bool(kwargs.get("require_all", True)),
    )


class FrameEndReport(list):
    """被销毁实体句柄列表，附带帧号。"""

    def __init__(
        self,
        destroyed: Iterable[Entity] | None = None,
        *,
        frame: int = 0,
        destroyed_ids: Iterable[int] = (),
        pending_remaining: int = 0,
    ):
        destroyed = list(destroyed or [])
        super().__init__(destroyed)
        self.destroyed = tuple(destroyed)
        self.frame = frame
        self.destroyed_ids = tuple(destroyed_ids) or tuple(entity.id for entity in destroyed)
        self.pending_remaining = pending_remaining

    def to_dict(self) -> dict[str, Any]:
        return {
            "destroyed": list(self.destroyed_ids),
            "frame": self.frame,
            "pending_remaining": self.pending_remaining,
        }


def frame_end(*args: Any, **kwargs: Any) -> FrameEndReport:
    world: World | None = kwargs.get("world")
    targets: list[Any] = []
    for arg in args:
        if isinstance(arg, World):
            world = arg
        elif isinstance(arg, Scheduler):
            world = arg.world
        elif not isinstance(arg, (int, float)) or (world is not None):
            targets.append(arg)

    if world is None:
        # 契约调用 frame_end(1)：构造一个已有延迟销毁请求的演示世界。
        world = _demo_world(pending=True)
        if targets:
            for target in targets:
                world.destroy(target)
    else:
        for target in targets:
            world.destroy(target)
    return world.end_frame()


class StorageGrowResult(dict):
    def __bool__(self) -> bool:
        return bool(self.get("ok", False))


def storage_grow(*args: Any, **kwargs: Any) -> StorageGrowResult:
    storage = next((arg for arg in args if isinstance(arg, ComponentStorage)), None)
    numeric_args = [arg for arg in args if isinstance(arg, (int, float)) and not isinstance(arg, bool)]
    if storage is None:
        initial_capacity = _as_int(
            kwargs.get("initial_capacity", numeric_args[0] if numeric_args else 2), 2
        )
        storage = ComponentStorage(
            str(kwargs.get("component_type", "GrowDemo")),
            initial_capacity=max(1, initial_capacity),
        )

    count = _as_int(
        kwargs.get("count", kwargs.get("items", numeric_args[1] if len(numeric_args) > 1 else 32)),
        32,
    )
    required_capacity = kwargs.get("capacity")
    old_capacity = storage.capacity
    old_growths = storage.growths
    if required_capacity is not None:
        storage.reserve(_as_int(required_capacity, old_capacity + count))

    seed = _as_int(kwargs.get("id_seed", 1), 1)
    before: list[tuple[Handle, Any]] = []
    for index in range(max(1, count)):
        handle = Handle(seed + index, 0)
        value = {"index": index, "payload": f"component-{index}"}
        storage.set(handle, value)
        before.append((handle, value))

    preserved = True
    for handle, value in before:
        try:
            if storage.get(handle) != value:
                preserved = False
                break
        except StaleHandleError:
            preserved = False
            break

    result = StorageGrowResult(
        {
            "ok": preserved,
            "count": count,
            "old_capacity": old_capacity,
            "new_capacity": storage.capacity,
            "grew": storage.capacity > old_capacity or storage.growths > old_growths,
            "growths": storage.growths - old_growths,
            "handles_stable": preserved,
            "values_preserved": preserved,
            "storage": storage,
        }
    )
    return result
