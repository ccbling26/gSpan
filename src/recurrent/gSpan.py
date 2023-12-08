from collections import defaultdict
from copy import copy
from itertools import count
from typing import List, Optional


class Edge:
    def __init__(self, eid: int, source: int, target: int, label: str) -> None:
        self.eid = eid
        self.source = source
        self.target = target
        self.label = label

    def __eq__(self, other: 'Edge') -> bool:
        return self.label == other.label

    def __le__(self, other: 'Edge') -> bool:
        return self.label <= other.label
    
    def __lt__(self, other: 'Edge') -> bool:
        return self.label < other.label
    
    def __ne__(self, other: 'Edge') -> bool:
        return self.label != other.label

    def __ge__(self, other: 'Edge') -> bool:
        return self.label >= other.label
    
    def __gt__(self, other: 'Edge') -> bool:
        return self.label > other.label

class Vertex:
    def __init__(self, vid: int, category: str, label: str) -> None:
        self.vid = vid
        self.category = category
        self.label = label
    
    def __eq__(self, other: 'Vertex') -> bool:
        return self.category == other.category and self.label == other.label

    def __le__(self, other: 'Vertex') -> bool:
        return (self.category, self.label) <= (other.category, other.label)

    def __lt__(self, other: 'Vertex') -> bool:
        return (self.category, self.label) < (other.category, other.label)
    
    def __ne__(self, other: 'Vertex') -> bool:
        return self.category != other.category or self.label != other.label

    def __ge__(self, other: 'Vertex') -> bool:
        return (self.category, self.label) >= (other.category, other.label)

    def __gt__(self, other: 'Vertex') -> bool:
        return (self.category, self.label) > (other.category, other.label)

class DG:
    """Directed Graph"""
    def __init__(self, gid: int) -> None:
        self.gid = gid
        self.vertices = {}
        self.edges = {}
    
    def add_vertex(self, vid: int, category: str, label: str):
        if vid in self.vertices:
            return
        vertex = Vertex(vid, category, label)
        self.vertices[vid] = vertex

    def add_edges(self, eid: int, source: int, target: int, label: str):
        if source not in self.vertices or target not in self.vertices:
            raise Exception("节点缺失")
        if (source, target) in self.edges:
            return
        edge = Edge(eid, source, target, label)
        self.edges[(source, target)] = edge
    
    def get_successors(self, source: int) -> List[int]:
        successors = []
        for k in self.edges:
            if k[0] == source:
                successors.append(k[1])
        return successors
    
    def get_edges(self, source: int) -> List[Edge]:
        successors = self.get_successors(source)
        edges = []
        for successor in successors:
            edges.append(self.edges[(source, successor)])
        return edges
    
class DFSEdge:
    def __init__(self, source: int, target: int, cllcl) -> None:
        """cllcl: a tuple, (source.category, source.label, edge.label, target.category, target.label)"""
        self.source = source
        self.target = target
        self.cllcl = cllcl

class DFSCode(list):
    def __init__(self) -> None:
        super(DFSCode, self).__init__()
        self.rightmost_path = []

    def to_graph(self, gid: int) -> DG:
        graph = DG(gid)
        for dfs_edge in self:
            dfs_edge: DFSEdge
            if dfs_edge.cllcl[0] != -1:
                graph.add_vertex(dfs_edge.source, dfs_edge.cllcl[0], dfs_edge.cllcl[1])
            if dfs_edge.cllcl[3] != -1:
                graph.add_vertex(dfs_edge.target, dfs_edge.cllcl[3], dfs_edge.cllcl[4])
            graph.add_edges(-1, dfs_edge.source, dfs_edge.target, dfs_edge.cllcl[2])
        return graph

    def build_rightmost_path(self):
        self.rightmost_path = []
        old_source = None
        # 从最后一个加入 DFSEdge 开始
        for i in range(len(self), -1, -1):
            dfs_edge: DFSEdge = self[i]
            source, target = dfs_edge.source, dfs_edge.target
            if source < target and (old_source is None or old_source == target):
                self.rightmost_path.append(i)
                old_source = source

    def count_vertices(self) -> int:
        vertices = set()
        for dfs_edge in self:
            dfs_edge: DFSEdge
            vertices.add(dfs_edge.source)
            vertices.add(dfs_edge.target)
        return len(vertices)

class ProjectItem:
    """频繁图和原始图的映射"""
    def __init__(self, gid: int, edge: Edge, prev: Optional['ProjectItem'] = None) -> None:
        self.gid = gid
        self.edge = edge
        self.prev = prev

class History:
    def __init__(self, project_item: ProjectItem) -> None:
        self.edges = []
        self.vertices_used = set()
        self.edges_used = set()
        while project_item:
            edge = project_item.edge
            self.edges.append(edge)
            self.vertices_used.add(edge.source)
            self.vertices_used.add(edge.target)
            self.edges_used.add(edge.eid)
            project_item = project_item.prev
        self.edges.reverse()

    def has_vertex(self, vid: int) -> bool:
        return vid in self.vertices_used

    def has_edge(self, eid: int) -> bool:
        return eid in self.edges_used

class Miner:
    def __init__(self, graphs: dict, min_support: int = 10, min_num_vertices: int = 1, max_num_vertices: float = float("inf")) -> None:
        self.graphs = graphs
        self.min_support = min_support
        if min_num_vertices > max_num_vertices:
            min_num_vertices, max_num_vertices = max_num_vertices, min_num_vertices
        self.min_num_vertices = min_num_vertices
        self.max_num_vertices = max_num_vertices

        self.one_node_frequent_subgraphs = []
        self.frequent_subgraphs = []
        self.gid_counter = count()
        self.dfs_code = DFSCode()

    
    def generate_one_node_frequent_subgraphs(self):
        """生成只有一个节点的频繁子图"""
        gid_vid_set = set()
        category_label_counter = defaultdict(int)
        for graph in self.graphs.values():
            graph: DG
            for vertex in graph.vertices.values():
                vertex: Vertex
                if (graph.gid, vertex.vid) not in gid_vid_set:
                    gid_vid_set.add((graph.gid, vertex.vid))
                    category_label_counter[(vertex.category, vertex.label)] += 1
        # 过滤不频繁的节点
        for (category, label), cnt in category_label_counter.items():
            if cnt >= self.min_support:
                graph = DG(gid=next(self.gid_counter))
                graph.add_vertex(0, category, label)
                self.one_node_frequent_subgraphs.append(graph)

    def run(self):
        if self.min_num_vertices <= 1:
            self.generate_one_node_frequent_subgraphs()
        if self.max_num_vertices <= 1:
            return
        root = defaultdict(list)
        for gid, graph in self.graphs.items():
            graph: DG
            for vid, source in graph.vertices.items():
                source: Vertex
                edges = graph.get_edges(vid)
                for edge in edges:
                    target: Vertex = graph.vertices[edge.target]
                    cllcl = (source.category, source.label, edge.label, target.category, target.label)
                    root[cllcl].append(ProjectItem(gid, edge, None))
        for cllcl, project_list in root.items():
            self.dfs_code.append(DFSEdge(0, 1, cllcl))
            self.subgraph_mining(project_list)
            self.dfs_code.pop()
    
    def subgraph_mining(self, project_list: List[ProjectItem]):
        graph_ids = len(set([item.gid for item in project_list]))
        # 过滤不频繁的子图
        if graph_ids < self.min_support:
            return
        if not self.is_min_dfs_code():
            return
        # 保存频繁子图（用 copy 还是 deepcopy？原代码看起来没有对底层数据进行修改）
        self.frequent_subgraphs.append(copy(self.dfs_code))
        
        num_vertices = self.dfs_code.count_vertices()
        self.dfs_code.build_rightmost_path()
        # index = 0 是最后加入的 dfs_edge
        rightmost_path = self.dfs_code.rightmost_path
        last_dfs_code: DFSEdge = self.dfs_code[rightmost_path[0]]
        max_target = last_dfs_code.target
        min_dfs_edge: DFSEdge = self.dfs_code[0]
        min_source = min_dfs_edge.source

        forward_root, backward_root = defaultdict(list), defaultdict(list)
        for project_item in project_list:
            graph: DG = self.graphs[project_item.gid]
            history = History(project_item)
            # 后向扩展，从最右路径的第一个节点开始
            for idx in rightmost_path[::-1]:
                dfs_edge: DFSEdge = self.dfs_code[idx]
                edge = self.get_backward_edge(graph, history.edges[idx], history.edges[rightmost_path[0]], history)
                if edge:
                    backward_root[(dfs_edge.source, edge.label)].append(ProjectItem(graph.gid, edge, project_item))
            # 节点数量约束
            if num_vertices >= self.max_num_vertices:
                continue
            # 前向扩展
            edges = self.get_pure_forward_edges(graph, history.edges[rightmost_path[0]], graph.vertices[min_source], history)
            for edge in edges:
                target: Vertex = graph.vertices[edge.target]
                forward_root[(max_target, edge.label, target.label)].append(ProjectItem(graph.gid, edge, project_item))
            for idx in rightmost_path:
                dfs_edge: DFSEdge = self.dfs_code[idx]
                source = dfs_edge.source
                edges = self.get_rightmost_path_forward_edges(graph, history[idx], graph.vertices[min_source], history)
                for edge in edges:
                    target: Vertex = graph.vertices[edge.target]
                    forward_root[(source, edge.label, target.label)].append(ProjectItem(graph.gid, edge, project_item))
        for (target, edge_label), val in backward_root.items():
            self.dfs_code.append(DFSEdge(max_target, target, (-1, edge_label, -1)))
            self.subgraph_mining(val)
            self.dfs_code.pop()
        for (source, edge_label, target_label), val in forward_root.items():
            self.dfs_code.append(DFSEdge(source, max_target + 1, (-1, edge_label, target_label)))
            self.subgraph_mining(val)
            self.dfs_code.pop()
    
    def is_min_dfs_code(self) -> bool:
        if len(self.dfs_code) == 1:
            return True
        graph = self.dfs_code.to_graph()
        min_dfs_code = DFSCode()
        root = defaultdict(list)
        for vid, vertex in graph.vertices.items():
            vertex: Vertex
            edges = graph.get_edges(vid)
            for edge in edges:
                target: Vertex = graph.vertices[edge.target]
                cllcl = (vertex.category, vertex.label, edge.label, target.category, target.label)
                root[cllcl].append(ProjectItem(graph.gid, edge, None))
        min_cllcl = min(root.keys())
        min_dfs_code.append(DFSEdge(0, 1, min_cllcl))

        def regenerate_min_dfs_code(project_list: List[ProjectItem]):
            nonlocal min_dfs_code
            nonlocal graph

            min_dfs_code.build_rightmost_path()
            rightmost_path = min_dfs_code.rightmost_path
            min_dfs_edge: DFSEdge = self.dfs_code[0]
            min_source = min_dfs_edge.source
            last_dfs_code: DFSEdge = self.dfs_code[rightmost_path[0]]
            max_target = last_dfs_code.target

            # 后向扩展边
            backward_root = defaultdict(list)
            flag, new_target = False, 0
            for idx in rightmost_path[::-1]:
                if flag:
                    break
                for project_item in project_list:
                    history = History(project_item)
                    edge = self.get_backward_edge(graph, history.edges[idx], history.edges[rightmost_path[0]], history)
                    if edge:
                        backward_root[edge.label].append(ProjectItem(graph.gid, edge, project_item))
                        new_target = min_dfs_code[idx].source
                        flag = True
            if flag:
                min_backward_edge_label = min(backward_root.keys())
                min_dfs_code.append(DFSEdge(max_target, new_target, (-1, min_backward_edge_label, -1)))
                idx = len(min_dfs_code) - 1
                if self.dfs_code[idx] != min_dfs_code[idx]:
                    return False
                return regenerate_min_dfs_code(backward_root[min_backward_edge_label])
            
            forward_root = defaultdict(list)
            flag, new_source = False, 0
            for project_item in project_list:
                history = History(project_item)
                edges = self.get_pure_forward_edges(graph, history.edges[rightmost_path[0]], graph.vertices[min_source], history)
                for edge in edges:
                    flag, new_source = True, max_target
                    target: Vertex = graph.vertices[edge.target]
                    forward_root[(edge.label, target.category, target.label)].append(ProjectItem(graph.gid, edge, project_item))
            for idx in rightmost_path:
                if flag:
                    break
                for project_item in project_list:
                    history = History(project_item)
                    edges = self.get_rightmost_path_forward_edges(graph, history.edges[idx], graph.vertices[min_source], history)
                    for edge in edges:
                        flag, new_source = True, min_dfs_code[idx].source
                        target: Vertex = graph.vertices[edge.target]
                        forward_root[(edge.label, target.category, target.label)].append(ProjectItem(graph.gid, edge, project_item))
            if not flag:
                return True
            min_forward_ecl = min(forward_root.keys())
            min_dfs_code.append(DFSEdge(new_source, max_target + 1, (-1, -1, *min_forward_ecl)))
            idx = len(min_dfs_code) - 1
            if self.dfs_code[idx] != min_dfs_code[idx]:
                return False
            return regenerate_min_dfs_code(forward_root[min_forward_ecl])

        return regenerate_min_dfs_code(root[min_cllcl])

    def get_backward_edge(self, graph: DG, edge1: Edge, edge2: Edge, history: History) -> Optional[Edge]:
        """检查 edge2 的 target 的后向边是否符合后向扩展规定，与 edge1 进行对比"""
        for edge in graph.get_edges(edge2.target):
            if history.has_edge(edge.eid) or edge.target != edge1.source:
                continue
            source = graph.vertices[edge1.source]
            target = graph.vertices[edge.target]
            if source < target or (source == target and edge1 <= edge):
                return edge
        return None

    def get_pure_forward_edges(self, graph: DG, edge: Edge, min_source: Vertex, history: History) -> List[Edge]:
        """在最右节点找前向扩展边"""
        result = []
        edges = graph.get_edges(edge.target)
        for e in edges:
            target: Vertex = graph.vertices[e.target]
            if min_source <= target and not history.has_vertex(e.target):
                result.append(e)
        return result

    def get_rightmost_path_forward_edges(self, graph: DG, edge: Edge, min_source: Vertex, history: History) -> List[Edge]:
        result = []
        target: Vertex = graph.vertices[edge.target]
        edges = graph.get_edges(edge.source)
        for e in edges:
            new_target: Vertex = graph.vertices[e.target]
            if history.has_vertex(e.target) or edge.target == e.target or min_source > new_target:
                continue
            if edge < e or (edge == e and target <= new_target):
                result.append(e)
        return result