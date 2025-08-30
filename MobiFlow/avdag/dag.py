from __future__ import annotations
from typing import Dict, List, Set, Tuple
from collections import defaultdict, deque

from .types import NodeSpec, TaskSpec

class DAG:
    def __init__(self, nodes: List[NodeSpec]):
        self.nodes: Dict[str, NodeSpec] = {n.id: n for n in nodes}
        # 统一的邻接关系（用于拓扑排序与无环校验）
        self.children: Dict[str, List[str]] = defaultdict(list)
        self.parents: Dict[str, List[str]] = defaultdict(list)
        # 来源细分：便于验证阶段分别按 AND/OR 语义处理
        self.parents_from_deps: Dict[str, List[str]] = defaultdict(list)
        self.parents_from_next: Dict[str, List[str]] = defaultdict(list)
        
        # 执行约束检查
        self._validate_dependencies_consistency(nodes)
        
        for n in nodes:
            for p in (n.deps or []):
                self.children[p].append(n.id)
                self.parents[n.id].append(p)
                self.parents_from_deps[n.id].append(p)
            # 根据 next 定义增加边：n -> succ
            for succ in (n.next or []):
                self.children[n.id].append(succ)
                self.parents[succ].append(n.id)
                self.parents_from_next[succ].append(n.id)
        # 验证无环
        self._assert_acyclic()

    def _assert_acyclic(self):
        indeg: Dict[str, int] = {nid: 0 for nid in self.nodes}
        for nid, ps in self.parents.items():
            indeg[nid] = len(ps)
        q = deque([nid for nid, d in indeg.items() if d == 0])
        seen = 0
        while q:
            cur = q.popleft()
            seen += 1
            for ch in self.children.get(cur, []):
                indeg[ch] -= 1
                if indeg[ch] == 0:
                    q.append(ch)
        if seen != len(self.nodes):
            raise ValueError("Graph contains a cycle")

    def _validate_dependencies_consistency(self, nodes: List[NodeSpec]):
        """验证 deps 和 next 定义的一致性，避免混淆和冗余"""
        import warnings
        
        # 构建 next 关系映射
        next_targets: Dict[str, List[str]] = defaultdict(list)
        for node in nodes:
            for succ in (node.next or []):
                next_targets[succ].append(node.id)
        
        issues = []
        
        for node in nodes:
            node_id = node.id
            has_deps = bool(node.deps)
            has_next_parents = bool(next_targets.get(node_id))
            
            # 检查1: 同时定义了 deps 和通过 next 被其他节点指向
            if has_deps and has_next_parents:
                next_parents = next_targets[node_id]
                deps_set = set(node.deps or [])
                next_set = set(next_parents)
                
                # 如果 deps 和 next 路径完全重复，这是冗余的
                if deps_set == next_set:
                    issues.append(f"节点 '{node_id}': deps {list(deps_set)} 与 next 路径来源 {list(next_set)} 完全重复，建议只使用 deps")
                # 如果 deps 和 next 路径部分重叠，可能导致混淆
                elif deps_set & next_set:
                    overlap = deps_set & next_set
                    issues.append(f"节点 '{node_id}': deps {list(deps_set)} 与 next 路径来源 {list(next_set)} 存在重叠 {list(overlap)}，可能导致语义混淆")
                # 如果完全不重叠，提示优先级
                else:
                    issues.append(f"节点 '{node_id}': 同时定义了 deps {list(deps_set)} 和 next 路径来源 {list(next_set)}，将优先使用 deps (AND 语义)")
            
            # 检查2: 节点同时定义了 deps 和 next（虽然技术上可行，但可能混淆）
            if has_deps and node.next:
                issues.append(f"节点 '{node_id}': 同时定义了 deps {node.deps} 和 next {node.next}，next 声明仅用于构建图结构")
        
        # 输出警告信息
        if issues:
            warning_msg = "DAG 依赖定义一致性警告：\n" + "\n".join(f"  - {issue}" for issue in issues)
            warnings.warn(warning_msg, UserWarning, stacklevel=3)
        
        # 检查3: 验证所有引用的节点都存在
        node_ids = {n.id for n in nodes}
        for node in nodes:
            # 检查 deps 引用
            for dep in (node.deps or []):
                if dep not in node_ids:
                    raise ValueError(f"节点 '{node.id}' 的 deps 引用了不存在的节点 '{dep}'")
            # 检查 next 引用
            for succ in (node.next or []):
                if succ not in node_ids:
                    raise ValueError(f"节点 '{node.id}' 的 next 引用了不存在的节点 '{succ}'")

    def topo_order(self) -> List[str]:
        indeg: Dict[str, int] = {nid: 0 for nid in self.nodes}
        for nid, ps in self.parents.items():
            indeg[nid] = len(ps)
        q = deque([nid for nid, d in indeg.items() if d == 0])
        order: List[str] = []
        while q:
            cur = q.popleft()
            order.append(cur)
            for ch in self.children.get(cur, []):
                indeg[ch] -= 1
                if indeg[ch] == 0:
                    q.append(ch)
        return order

    def sinks(self) -> List[str]:
        return [nid for nid in self.nodes if len(self.children.get(nid, [])) == 0]

    def get_all_paths_to_targets(self, target_nodes: List[str]) -> List[List[str]]:
        """获取从根节点到目标节点的所有可能路径"""
        all_paths = []
        
        # 找到所有根节点（无父节点的节点）
        root_nodes = [nid for nid in self.nodes if len(self.parents.get(nid, [])) == 0]
        
        def dfs_paths(current: str, path: List[str], visited: set):
            if current in visited:
                return  # 避免环路
            
            visited.add(current)
            path.append(current)
            
            # 如果当前节点是目标节点之一，记录路径
            if current in target_nodes:
                all_paths.append(path.copy())
            
            # 继续向子节点探索
            for child in self.children.get(current, []):
                dfs_paths(child, path, visited.copy())
            
            path.pop()
        
        # 从每个根节点开始探索
        for root in root_nodes:
            dfs_paths(root, [], set())
        
        return all_paths

    def log_possible_paths(self, success_nodes: List[str], logger):
        """输出配置中存在的可能路径到日志"""
        logger.info("=== DAG 路径分析 ===")
        
        # 输出节点依赖关系概览
        logger.debug("节点依赖关系:")
        for nid in self.topo_order():
            node = self.nodes[nid]
            deps_info = f"deps={node.deps}" if node.deps else "deps=None"
            next_info = f"next={node.next}" if node.next else "next=None"
            logger.debug(f"  {nid}: {deps_info}, {next_info}")
        
        # 输出父子关系
        logger.debug("父子关系:")
        for nid in self.nodes:
            deps_parents = self.parents_from_deps.get(nid, [])
            next_parents = self.parents_from_next.get(nid, [])
            if deps_parents:
                logger.debug(f"  {nid} <- {deps_parents} (deps, AND语义)")
            if next_parents:
                logger.debug(f"  {nid} <- {next_parents} (next, OR语义)")
        
        # 获取并输出所有可能路径
        all_paths = self.get_all_paths_to_targets(success_nodes)
        
        if all_paths:
            logger.info(f"发现 {len(all_paths)} 条可能的成功路径:")
            for i, path in enumerate(all_paths, 1):
                path_str = " -> ".join(path)
                logger.info(f"  路径 {i}: {path_str}")
        else:
            logger.info("未发现任何可能的成功路径")
        
        logger.info("=== 路径分析结束 ===\n")

__all__ = ["DAG"]
