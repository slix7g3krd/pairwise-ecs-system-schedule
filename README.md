# pairwise-ecs-system-schedule

## 领域

游戏、图形与多媒体

## 主题

ECS 系统调度与确定性

## 说明

读写集排序、延迟销毁与组件存储扩容。

初始状态下 `gameplay/ecs_system_schedule.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
