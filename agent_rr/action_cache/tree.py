from enum import Enum
import torch
import time
try:
    from omniparser.omniparser import Omniparser
except ImportError as e:
    print("Import omniparser failed, some features may not work")

from .reranker import Qwen3Reranker
from .embedder import Qwen3Embedder
from .action import Action, UIElement

EMBEDDER_THRESHOLD = 0.8
RERANKER_MIN_CONF = 0.75

class MatchMode(Enum):
    EXACT = 1
    FUZZY = 2

class Task:
    def __init__(self, description):
        self.description = description

    def __eq__(self, other):
        return self.description == other.description

    def __str__(self):
        return self.description

    def __repr__(self):
        return f"Task(description={self.description})"

class ActionTreeEdge:
    def __init__(self, action=None, tasks=[], to=None):
        self.action = action
        self.to = to
        self.tasks = tasks

    def add_task(self, task):
        self.tasks.append(task)

    def remove_task(self, task_idx):
        self.tasks.pop(task_idx)

    def __str__(self):
        return f"{self.action} {self.tasks}"

class ActionTreeEdgeFuzzy(ActionTreeEdge):
    def __init__(self, action=None, tasks=[], to=None, task_embeddings=[], keywords=[]):
        l = len(tasks)
        if l == 0:
            raise ValueError("Tasks list is empty")
        if l != task_embeddings.shape[0]:
            raise ValueError("Tasks list length must match task_embeddings length")
        if l != len(keywords):
            raise ValueError("Tasks list length must match keywords length")
        super().__init__(action, tasks, to)
        self.task_embeddings = task_embeddings
        self.keywords = keywords

    def add_task(self, task, task_embedding, keyword=""):
        self.tasks.append(task)
        self.task_embeddings = torch.cat([self.task_embeddings, task_embedding], dim=0)
        self.keywords.append(keyword)

    def remove_task(self, task_idx):
        self.tasks.pop(task_idx)
        self.task_embeddings = torch.cat([self.task_embeddings[:task_idx], self.task_embeddings[task_idx+1:]], dim=0)
        self.keywords.pop(task_idx)

    def reset_keyword(self, keyword):
        for i, kw in enumerate(self.keywords):
            if kw == keyword:
                self.keywords[i] = ""

    def __str__(self):
        return f"{super().__str__()} {self.keywords}"

class SuperNode:
    def __init__(self, nodes):
        self.nodes = nodes

    def add_node(self, node):
        self.nodes.append(node)

class ShortCutCheckResult(Enum):
    MATCH_INTERMEDIATE = 1
    MATCH_SECOND_LAST = 2
    MATCH_LAST = 3
    NOT_MATCH = 4

class ShortCutTemplate:
    def __init__(self, action_names, last_action):
        self.action_names = action_names
        self.last_action = last_action

    def check(self, action, step):
        if step >= len(self.action_names):
            return ShortCutCheckResult.NOT_MATCH
        if step == len(self.action_names):
            if action == self.last_action:
                return ShortCutCheckResult.MATCH_LAST
        else:
            if action.name == self.action_names[step]:
                if step == len(self.action_names) - 1:
                    return ShortCutCheckResult.MATCH_SECOND_LAST
                else:
                    return ShortCutCheckResult.MATCH_INTERMEDIATE
        return ShortCutCheckResult.NOT_MATCH

class ShortCut:
    def __init__(self, split_node, template, supernode):
        self.split_node = split_node
        self.template = template
        self.supernode = supernode

    def check(self, action, step):
        return self.template.check(action, step)

class ActionTreeNode:
    def __init__(self, parent=None):
        self.edges = []
        self.parent = parent
        self.parent_edge_idx = None
        self.screenshot = None
        # if a node is a possible split node, pin it
        self.split_pin = False
        if parent is not None:
            self.depth = parent.depth + 1
        else:
            self.depth = 0

    def add_child(self, action, task):
        for e in self.edges:
            # merge happens here
            if e.action == action:
                e.add_task(task)
                return e.to
        new_node = ActionTreeNode(self)
        new_edge = ActionTreeEdge(action, [task], new_node)
        new_node.parent_edge_idx = len(self.edges)
        self.edges.append(new_edge)
        return new_node

    def get_cached_action(self, task):
        ret = []
        for e in self.edges:
            for t in e.tasks:
                if t == task:
                    ret.append((e.action, e.to))
        return ret

    def _remove_task(self, task):
        next_node = None
        for edge_idx, e in enumerate(self.edges):
            for task_idx, t in enumerate(e.tasks):
                if t == task:
                    e.remove_task(task_idx)
                    next_node = e.to
                    break
            if len(e.tasks) == 0:
                self.edges.pop(edge_idx)
                return None
            elif next_node is not None:
                return next_node
        return None

    def remove_task_trace(self, task):
        node = self
        while node is not None:
            node = node._remove_task(task)

    def get_incoming_edge(self):
        return self.parent.edges[self.parent_edge_idx]

    def get_incoming_action(self):
        return self.get_incoming_edge().action

    def remove_child(self, child):
        if child.parent is not self:
            raise ValueError("Not a child of this node")
        self.edges.pop(child.parent_edge_idx)

    def try_find_shortcuts(self):
        # assume self is the split node, find the possible merged supernodes

        # TODO: use values from config
        min_supernode_capacity = 2
        min_shortcut_len, max_shortcut_len = 2, 3

        def _can_merge_to_supernode(nodes):
            # check incoming edges
            if len(nodes) < min_supernode_capacity:
                return False
            action = None
            for n in nodes:
                if action is None:
                    action = n.get_incoming_action()
                elif action != n.get_incoming_action():
                    return False
            return True

        def _have_same_parent(nodes):
            parent = nodes[0].parent
            for n in nodes[1:]:
                if n.parent is not parent:
                    return False
            return True

        def _dfs(nodes, trace, supernodes, templates):
            cur_len = len(trace)
            if cur_len > 1 and _have_same_parent(nodes):
                return
            if cur_len > max_shortcut_len:
                return
            if cur_len >= min_shortcut_len and _can_merge_to_supernode(nodes):
                # print(nodes)
                supernodes.append(SuperNode(nodes))
                action_names = trace[:-1]
                last_action = nodes[0].get_incoming_action()
                templates.append(ShortCutTemplate(action_names, last_action))
                # greedy match for minimizing shortcut length
                return
            next_layer_nodes = []
            next_actions = []
            for n in nodes:
                for e in n.edges:
                    next_layer_nodes.append(e.to)
                    next_actions.append(e.action)
            # group next_layer_nodes by action
            action_group = {}
            for n, a in zip(next_layer_nodes, next_actions):
                if a.name not in action_group:
                    action_group[a.name] = []
                action_group[a.name].append(n)
            for action_name, group in action_group.items():
                next_trace = trace + [action_name]
                _dfs(group, next_trace, supernodes, templates)

        supernodes = []
        templates = []
        trace = []
        _dfs([self], trace, supernodes, templates)
        shortcuts = [ShortCut(self, t, s) for t, s in zip(templates, supernodes)]
        return shortcuts

class ActionTreeNodeFuzzy(ActionTreeNode):
    def __init__(self, parent=None):
        super().__init__(parent)

    def add_child(self, action, task, task_embedding):
        keyword = self._extract_keyword(task, action)
        for e in self.edges:
            # merge happens here
            if e.action == action:
                e.add_task(task, task_embedding, keyword)
                return e.to
        new_node = ActionTreeNodeFuzzy(self)
        new_edge = ActionTreeEdgeFuzzy(action, [task], new_node, task_embedding, [keyword])
        new_node.parent_edge_idx = len(self.edges)
        self.edges.append(new_edge)
        return new_node

    def _extract_keyword(self, task, action):
        return ""

    def get_cached_action(self, task, step_embedding):
        ret = []
        for e in self.edges:
            hit = util.semantic_search(step_embedding, e.task_embeddings, top_k=1, score_function=util.dot_score)[0]
            # print(hit)
            score = hit[0]['score']
            if score < EMBEDDER_THRESHOLD:
                continue
            corpus_id = hit[0]['corpus_id']
            keyword = e.keywords[corpus_id]
            if keyword not in task.description:
                continue
            hit_task = e.tasks[corpus_id]
            print(hit_task, score)
            ret.append((e.action, e.to, keyword, hit_task))
        return ret

    def reset_keyword(self, keyword):
        for e in self.edges:
            e.reset_keyword(keyword)


class ActionTree:
    def __init__(self,
                 env,
                 agent,
                 action_class=Action,
                 done=lambda a: a.name == 'END',
                 mode: MatchMode = MatchMode.EXACT,
                 embedder_config=None,
                 reranker_config=None,
                 enable_ui_detection=False,
                 omniparser_config=None):
        self.env = env
        self.agent = agent
        self.done = done
        self.mode = mode
        self.action_class = action_class
        self.enable_ui_detection = enable_ui_detection
        self.generate_only = False
        self.shortcuts = []
        self.num_tasks_last_check = 0
        if mode == MatchMode.EXACT:
            self.embedder = None
            self.root = ActionTreeNode()
        elif mode == MatchMode.FUZZY:
            if embedder_config is None:
                raise ValueError("embedder_config is required for fuzzy matching")
            self.embedder = Qwen3Embedder(embedder_config)
            self.root = ActionTreeNodeFuzzy()

            if reranker_config is not None:
                self.reranker = Qwen3Reranker(reranker_config)
            else:
                self.reranker = None
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if omniparser_config is not None:
            self.enable_ui_detection = True
            self.omniparser = Omniparser(omniparser_config)
        else:
            self.omniparser = None

        self.reset_counter()

    def reset_counter(self):
        self.env_counter = 0.0
        self.inference_counter = 0.0
        self.detection_counter = 0.0
        self.embedding_counter = 0.0

    def print_counter(self):
        print(f"env_counter: {self.env_counter}, inference_counter: {self.inference_counter}, detection_counter: {self.detection_counter}, embedding_counter: {self.embedding_counter}")

    def clear(self):
        self.shortcuts = []
        self.num_tasks_last_check = 0
        self.root = ActionTreeNode() if self.mode == MatchMode.EXACT else ActionTreeNodeFuzzy()

    def target_elem_changed(self, cur_screen, action):
        if action.target_elem is None:
            return False
        if cur_screen is None:
            return False
        target_elem = action.target_elem
        bbox = target_elem.bbox
        x1, x2 = map(lambda x: x / 1000 * cur_screen.width, (bbox[0], bbox[2]))
        y1, y2 = map(lambda x: x / 1000 * cur_screen.height, (bbox[1], bbox[3]))
        cropped_screen = cur_screen.crop((x1, y1, x2, y2))
        if self.omniparser is None:
            new_elem = UIElement(bbox, target_elem.content, cropped_screen)
            return new_elem != target_elem
        else:
            parsed_elems = self.omniparser.parse(cropped_screen)

            for elem in parsed_elems:
                if elem["content"] == target_elem.content:
                    return False
            return True

    def get_num_tasks(self):
        return sum([len(e.tasks) for e in self.root.edges])

    def generate_shortcuts(self):
        # periodically check if there are new shortcuts
        # use bfs
        queue = [self.root]
        self.shortcuts = []
        while queue:
            node = queue.pop(0)
            for e in node.edges:
                queue.append(e.to)
            if node is self.root:
                continue
            shortcuts = node.try_find_shortcuts()
            # last_action cannot be done action
            shortcuts = [sc for sc in shortcuts if not self.done(sc.template.last_action)]
            self.shortcuts.extend(shortcuts)
            node.split_pin = shortcuts != []

    def execute(self, task_description):
        node = self.root
        history = []
        task = Task(task_description)
        if self.mode == MatchMode.FUZZY:
            start_time = time.time()
            num_precomute = 16
            step_embeddings = self.embedder.embed([task_description] * num_precomute, steps=range(1, num_precomute + 1))
            end_time = time.time()
            self.embedding_counter += end_time - start_time
            recompute_times = 0

        tracking_shortcut = False
        shortcut_action = None

        while True:
            # candidate (action, next_node) pairs
            action_nodes = []
            depth = node.depth

            start_time = time.time()

            if self.mode == MatchMode.EXACT:
                if shortcut_action is not None:
                    shortcut_next_node = node.add_child(shortcut_action, task)
                    action_nodes = [(shortcut_action, shortcut_next_node)]
                    keywords = [shortcut_next_node.get_incoming_edge().keywords[-1]]
                    shortcut_action = None
                else:
                    action_nodes = node.get_cached_action(task)
            else:
                if depth >= (recompute_times + 1) * num_precomute:
                    recompute_times += 1
                    step_embeddings = self.embedder.embed(
                        [task_description] * num_precomute,
                        steps=range(recompute_times * num_precomute + 1, (recompute_times + 1) * num_precomute + 1)
                    )

                step_embedding = step_embeddings[depth - recompute_times * num_precomute].unsqueeze(0)

                if shortcut_action is not None:
                    shortcut_next_node = node.add_child(shortcut_action, task, step_embedding)
                    action_nodes = [(shortcut_action, shortcut_next_node)]
                    keywords = [shortcut_next_node.get_incoming_edge().keywords[-1]]
                    shortcut_action = None
                else:
                    action_node_keyword_tasks = node.get_cached_action(task, step_embedding)
                    hit_tasks = [t.description for a, n, kw, t in action_node_keyword_tasks]
                    if len(action_node_keyword_tasks) == 0:
                        print(f"No similar task found.")
                    else:
                        print(f"Found similar task: {hit_tasks}")
                    if self.reranker is not None and len(hit_tasks) > 0:
                        scores = self.reranker.rerank(query_tasks=hit_tasks, document_task=task_description, step=depth + 1)
                        indices = [i for i, score in enumerate(scores) if score > RERANKER_MIN_CONF]
                        action_node_keyword_tasks = [action_node_keyword_tasks[i] for i in indices]
                        if len(indices) != len(hit_tasks):
                            print(f"Reranker filtered tasks: {[hit_tasks[i] for i in range(len(hit_tasks)) if i not in indices]}")
                    action_nodes = [(a, n) for a, n, kw, t in action_node_keyword_tasks]
                    keywords = [kw for a, n, kw, t in action_node_keyword_tasks]
            end_time = time.time()
            self.embedding_counter += end_time - start_time

            if node.split_pin and not self.generate_only and not tracking_shortcut:
                # start tracking possible shortcut
                print("Start tracking shortcut")
                tracking_shortcut = True
                possible_shortcuts = [sc for sc in self.shortcuts if sc.split_node is node]
                cur_step = 0

            # check if the action needs to be generated by model, or we can use cached action
            needs_generation = len(action_nodes) == 0 or self.generate_only

            start_time = time.time()
            agent_input = self.env.get_agent_input(history, task_description)
            end_time = time.time()
            self.env_counter += end_time - start_time

            screenshot = agent_input.get("image", None)
            # if UI changed, we need to generate the action
            if not needs_generation:
                if self.enable_ui_detection:
                    start_time = time.time()
                    for i, (a, n) in enumerate(action_nodes):
                        if not self.target_elem_changed(screenshot, a):
                            action = a
                            next_node = n
                            if self.mode == MatchMode.FUZZY:
                                keyword = keywords[i]
                            break
                    # the else block is executed if the for loop is not broken
                    else:
                        print("warning: target element changed")
                        needs_generation = True

                    end_time = time.time()
                    self.detection_counter += end_time - start_time
                else:
                    action, next_node = action_nodes[0]
                    if self.mode == MatchMode.FUZZY:
                        keyword = keywords[0]

            if needs_generation:
                print("Cache miss")
                start_time = time.time()
                agent_output = self.agent.generate(agent_input)
                end_time = time.time()
                self.inference_counter += end_time - start_time
                action = self.action_class(**agent_output)
                # extract target element and store it in action
                if self.enable_ui_detection:
                    start_time = time.time()
                    action.extract_target_elem(screenshot, self.omniparser)
                    end_time = time.time()
                    self.detection_counter += end_time - start_time
                if self.mode == MatchMode.EXACT:
                    next_node = node.add_child(action, task)
                else:
                    next_node = node.add_child(action, task, step_embedding)
            else:
                print("Cache hit")
                edge = next_node.get_incoming_edge()
                # only add similar task to the edge
                if self.mode == MatchMode.FUZZY and task not in edge.tasks:
                    edge.add_task(task, step_embedding, keyword)

            if tracking_shortcut:
                new_possible_shortcuts = []
                for i, sc in enumerate(possible_shortcuts):
                    check_result =  sc.check(action, cur_step)
                    if check_result == ShortCutCheckResult.MATCH_SECOND_LAST:
                        # can use cached action in next iteration
                        tracking_shortcut = False
                        # add a child for next_node in advance
                        # in next iteration, cache hit is guaranteed
                        if needs_generation:
                            last_action = sc.template.last_action
                            shortcut_action = last_action
                        break
                    if check_result == ShortCutCheckResult.MATCH_INTERMEDIATE:
                        new_possible_shortcuts.append(sc)
                else:
                    cur_step += 1
                    possible_shortcuts = new_possible_shortcuts
                    if len(possible_shortcuts) == 0:
                        tracking_shortcut = False

            # execute the action
            if self.done(action):
                break
            history.append(action)

            start_time = time.time()
            self.env.execute(action)
            end_time = time.time()
            self.env_counter += end_time - start_time

            node = next_node

        # periodically generate shortcuts
        if not self.generate_only:
            period = 1
            num_tasks = self.get_num_tasks()
            if num_tasks - self.num_tasks_last_check >= period:
                self.num_tasks_last_check = num_tasks
                self.generate_shortcuts()
                print(f"number of shortcuts: {len(self.shortcuts)}")
                for sc in self.shortcuts:
                    print(f"split_node: {sc.split_node}, template: {sc.template.action_names}, last_action: {sc.template.last_action}, supernode size: {len(sc.supernode.nodes)}")
