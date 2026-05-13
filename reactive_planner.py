import py_trees
import json
import doubao
from bt_nodes import SkillAction, VLMVerifier
from py_trees.common import Status
from grasp_fastapi import (
    get_camera_image, detect, cloud, create_grasp, move_pregrasp, 
    move_grasp, gripper, vertical, execute_grasp_2, execute_init,
    task_list, content_list_all, target
)

# 动作映射表：将 VLM 返回的字符串映射到具体函数
ACTION_MAP = {
    "camera-image": get_camera_image,
    "detect-object": detect, # 注意参数处理
    "create-cloud": cloud,
    "create-grasp": create_grasp,
    "move-pregrasp": move_pregrasp,
    "move-grasp": move_grasp,
    "vertical-grasp": vertical,
    "gripper-action": gripper, # 注意参数 flag
    "execute-grasp2": execute_grasp_2,
    "execute-init": execute_init
}

class ReactiveController:
    def __init__(self):
        self.root = None
        self.current_target = "apple" # 示例目标

    def build_main_tree(self):
        """
        构建标准的主流程行为树 (对应论文的 "Pre-Execution Plan")
        Sequence:
          -> Detect -> Grasp Plan -> PreGrasp -> Grasp -> Lift -> Place
        """
        root = py_trees.composites.Sequence("MainTask", memory=True)

        # 1. 感知阶段
        perception = py_trees.composites.Sequence("Perception")
        perception.add_child(SkillAction("GetImage", get_camera_image))
        perception.add_child(SkillAction("Detect", detect, target_class=self.current_target))
        perception.add_child(SkillAction("Cloud", cloud))
        perception.add_child(SkillAction("PlanGrasp", create_grasp))
        
        # 2. 抓取执行阶段 (带 VLM 验证)
        execution = py_trees.composites.Sequence("Execution")
        
        # 预抓取
        execution.add_child(SkillAction("MovePreGrasp", move_pregrasp))
        execution.add_child(VLMVerifier("CheckPreGrasp", "移动到预抓取位置"))
        
        # 抓取
        execution.add_child(SkillAction("MoveGrasp", move_grasp))
        execution.add_child(VLMVerifier("CheckGraspPose", "移动到抓取位置"))
        
        # 闭合夹爪
        execution.add_child(SkillAction("CloseGripper", gripper, flag=1))
        execution.add_child(VLMVerifier("CheckClosed", "夹爪闭合"))
        
        # 提升
        execution.add_child(SkillAction("Lift", vertical))
        execution.add_child(VLMVerifier("CheckLift", "提升物体"))

        root.add_child(perception)
        root.add_child(execution)
        self.root = root

    def run_reactive_loop(self):
        """
        对应论文的 "Real-time Monitoring" 循环
        如果树执行失败，触发 "Reactive Planner" (doubao.fault_recover)
        """
        self.build_main_tree()
        behavior_tree = py_trees.trees.BehaviourTree(self.root)
        
        # 建立 Blackboard 供节点共享数据
        # blackboard = py_trees.blackboard.Client(name="Controller")
        # blackboard.register_key(key="target_object", access=py_trees.common.Access.WRITE)
        # blackboard.target_object = self.current_target

        print("=== 开始执行任务 (Behavior Tree) ===")
        behavior_tree.tick()

        while behavior_tree.root.status != Status.SUCCESS:
            
            # 1. 正常执行 Tick
            behavior_tree.tick()
            
            # 2. 监控是否失败 (Failure Handling)
            if behavior_tree.root.status == Status.FAILURE:
                print("\n[ALERT] 任务执行失败，激活反应式规划器...")
                
                # --- 论文核心：Reactive Planner ---
                # 调用 VLM 分析原因并生成恢复序列
                recovery_plan_json = doubao.fault_recover(
                    content_list_all, # 传入历史图像
                    task_list,        # 传入历史任务状态
                    target=self.current_target
                )
                
                try:
                    # 解析 VLM 返回的 JSON 计划
                    # 假设 doubao 返回的是 string，需要 json.loads
                    # 注意：要在 prompt 中严格限制只返回 json 数组
                    recovery_actions = json.loads(recovery_plan_json)
                    print(f"[RECOVERY PLAN] VLM 生成补救策略: {recovery_actions}")
                    
                    # 执行恢复策略
                    success = self.execute_recovery_sequence(recovery_actions)
                    
                    if success:
                        print("[RECOVERED] 恢复成功，重置树状态并继续...")
                        # 恢复成功后，我们可以选择重置树，或者从特定节点继续
                        # 简单起见，这里选择重置相关节点状态或重建树
                        # 在 py_trees 中，可以用 visitor 模式重置
                        behavior_tree.root.stop(Status.INVALID) 
                        # 清空 task_list 以免干扰下一次判断? (视具体逻辑而定)
                        continue
                    else:
                        print("[FATAL] 恢复策略执行失败，任务终止。")
                        break
                        
                except Exception as e:
                    print(f"解析恢复计划失败: {e}")
                    break
            
            # 简单模拟循环间隔
            import time
            time.sleep(0.1)

    def execute_recovery_sequence(self, actions):
        """
        执行 VLM 生成的动态 JSON 动作序列
        """
        for step in actions:
            action_name = step.get("action")
            params = step.get("parameters", {})
            
            print(f"  -> 执行恢复动作: {action_name} | 参数: {params}")
            
            if action_name in ACTION_MAP:
                func = ACTION_MAP[action_name]
                try:
                    # 动态调用函数
                    # 注意：这里需要处理 async/sync 问题，同 SkillAction
                    import asyncio
                    loop = asyncio.get_event_loop()
                    if asyncio.iscoroutinefunction(func):
                        res = loop.run_until_complete(func(**params))
                    else:
                        res = func(**params)
                    
                    # 将恢复动作的执行结果也加入 task_list，让 VLM 知道进度
                    task_list.append(f"{action_name}:SUCCESS")
                    
                except Exception as e:
                    print(f"  -> 恢复动作 {action_name} 失败: {e}")
                    task_list.append(f"{action_name}:FAILURE")
                    return False
            else:
                print(f"  -> 未知动作: {action_name}")
                return False
        return True