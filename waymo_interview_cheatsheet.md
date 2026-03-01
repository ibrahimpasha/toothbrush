# Waymo Staff SWE Coding Interview Cheat Sheet

---

## PAGE 1 — Core Patterns & Templates

### Complexity Quick Reference
| Structure     | Access    | Search    | Insert    | Delete    |
|---------------|-----------|-----------|-----------|-----------|
| Array         | O(1)      | O(n)      | O(n)      | O(n)      |
| Hash Map      | —         | O(1)      | O(1)      | O(1)      |
| Heap          | O(1) top  | O(n)      | O(log n)  | O(log n)  |
| BST (balanced)| O(log n)  | O(log n)  | O(log n)  | O(log n)  |
| Graph BFS/DFS | —         | O(V+E)    | —         | —         |

---

### Two Pointers
**Use when:** sorted array, palindrome, pair sum, removing duplicates

```
arr = [1, 3, 5, 7, 9, 11]   target = 14
       l                r    sum=12 < 14  → l++
          l             r    sum=14 ✓
```
```python
def two_sum_sorted(arr, target):
    l, r = 0, len(arr) - 1
    while l < r:
        s = arr[l] + arr[r]
        if s == target: return [l, r]
        elif s < target: l += 1
        else: r -= 1
```

---

### Sliding Window
**Use when:** subarray/substring with constraint, max/min window

```
Fixed window k=3:
 arr = [1, 3, -1, -3,  5,  3]
       [_____]           sum=3
          [_____]        sum=-1
              [_____]    sum=1
                  [_____]sum=5 ← max

Variable window (longest unique substring):
 s = "abcabc"
      l r          expand r while unique
      l        r   'a' seen → shrink l
```
```python
def max_subarray_k(arr, k):
    window, res = sum(arr[:k]), sum(arr[:k])
    for i in range(k, len(arr)):
        window += arr[i] - arr[i - k]
        res = max(res, window)
    return res

def longest_unique(s):
    seen, l, res = {}, 0, 0
    for r, c in enumerate(s):
        if c in seen: l = max(l, seen[c] + 1)
        seen[c] = r
        res = max(res, r - l + 1)
    return res
```

---

### Binary Search
**Use when:** sorted data, "find minimum that satisfies X", rotated arrays

```
arr = [1, 3, 5, 7, 9, 11, 13]  target=9
       l           m         r  arr[m]=7 < 9 → l=m+1
                   l    m    r  arr[m]=11> 9 → r=m-1
                   l  r         arr[l]=9 ✓

Search on answer space:
  lo ————————————————— hi
       feasible? NO  YES
                 ↑ boundary = answer
```
```python
def bs(arr, target):
    l, r = 0, len(arr) - 1
    while l <= r:
        m = (l + r) // 2
        if arr[m] == target: return m
        elif arr[m] < target: l = m + 1
        else: r = m - 1
    return -1

def search_on_answer(lo, hi, feasible):
    while lo < hi:
        mid = (lo + hi) // 2
        if feasible(mid): hi = mid
        else: lo = mid + 1
    return lo
```

---

### BFS (Shortest Path / Level Order)
**Use when:** shortest path unweighted, level-by-level, multi-source

```
      1        ← level 0   Queue: [1]
    /   \
   2     3     ← level 1   Queue: [2, 3]
  / \     \
 4   5     6   ← level 2   Queue: [4, 5, 6]

Grid BFS (4-dir):
  S . . #       S=start, G=goal, #=wall
  . # . .       BFS radiates outward uniformly
  . # . G       guarantees shortest path
  . . . .
```
```python
from collections import deque

def bfs(grid, start):
    rows, cols = len(grid), len(grid[0])
    q = deque([(start[0], start[1], 0)])
    visited = {start}
    while q:
        r, c, dist = q.popleft()
        if (r, c) == goal: return dist
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0<=nr<rows and 0<=nc<cols and (nr,nc) not in visited:
                visited.add((nr,nc))
                q.append((nr, nc, dist+1))
```

---

### DFS (Backtracking / Connected Components)
**Use when:** all paths, permutations, combinations, flood fill

```
Tree DFS (pre-order):          Call stack trace:
      A                        dfs(A)
     / \                         dfs(B)
    B   C                          dfs(D) → leaf, return
   / \                           dfs(E) → leaf, return
  D   E                        dfs(C) → leaf, return
Visit order: A B D E C

Backtracking (subsets of [1,2,3]):
path=[]  → choose 1 → path=[1] → choose 2 → [1,2] ✓
                                → choose 3 → [1,3] ✓
                     → undo 1, choose 2 → [2] → choose 3 → [2,3] ✓
```
```python
def dfs(graph, node, visited=None):
    if visited is None: visited = set()
    visited.add(node)
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)

def backtrack(path, choices):
    if is_solution(path):
        results.append(path[:])
        return
    for c in choices:
        path.append(c)
        backtrack(path, remaining_choices)
        path.pop()
```

---

### Dijkstra (Weighted Shortest Path)
**Use when:** weighted graph, min cost path — core to AV routing!

```
Graph:                  Processing order (min-heap):
  A —2→ B —3→ D         pop(0,A): relax B=2, C=4
  |    ↗      |          pop(2,B): relax D=5, C=min(4,6)=4
  4  (1)      1          pop(4,C): relax B=min(2,5)=2 (skip)
  ↓ /         ↓          pop(5,D): done
  C ——5→———→ E
                          dist: A=0, B=2, C=4, D=5, E=6
```
```python
import heapq

def dijkstra(graph, src):
    dist = {src: 0}
    pq = [(0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, float('inf')): continue
        for v, w in graph[u]:
            nd = d + w
            if nd < dist.get(v, float('inf')):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist
```

---

## PAGE 2 — DP, Graphs, Intervals & Waymo-Specific Tips

---

### Dynamic Programming
**Identify:** optimal substructure + overlapping subproblems
**Approach:** define state → recurrence → base case → fill order

```
Coin Change (coins=[1,3,4], amount=6):
  idx:  0  1  2  3  4  5  6
  dp: [ 0, 1, 2, 1, 1, 2, 2 ]
                  ↑        ↑
              dp[3]=1    dp[6]=2 (3+3)

LCS table (s1="ABCB", s2="BCAB"):
       ""  B  C  A  B
    ""  0  0  0  0  0
    A   0  0  0  1  1
    B   0  1  1  1  2
    C   0  1  2  2  2
    B   0  1  2  2  3  ← answer
```
```python
def coin_change(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for a in range(1, amount + 1):
        for c in coins:
            if c <= a: dp[a] = min(dp[a], dp[a-c] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1

def lcs(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]: dp[i][j] = dp[i-1][j-1] + 1
            else: dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

def knapsack(weights, values, W):
    dp = [0] * (W + 1)
    for w, v in zip(weights, values):
        for cap in range(W, w-1, -1):  # reverse for 0/1
            dp[cap] = max(dp[cap], dp[cap-w] + v)
    return dp[W]
```

---

### Intervals
**Use when:** scheduling, merging, overlaps

```
Merge intervals:
  [1——3]                A
       [2————5]         B  overlaps A → merge
              [6——8]    C  no overlap → new
                 [7—9]  D  overlaps C → merge

  Result: [1————5]  [6————9]

Sweep line (min meeting rooms):
  ────────────────────────────→ time
  [===A===]
       [===B===]    rooms=2 here (both active)
              [=C=]
  events: (+1,+1,-1,+1,-1,-1) → max running sum = 2
```
```python
def merge_intervals(intervals):
    intervals.sort()
    res = [intervals[0]]
    for s, e in intervals[1:]:
        if s <= res[-1][1]: res[-1][1] = max(res[-1][1], e)
        else: res.append([s, e])
    return res

def can_attend(intervals):
    intervals.sort()
    for i in range(1, len(intervals)):
        if intervals[i][0] < intervals[i-1][1]: return False
    return True
```

---

### Union-Find (Disjoint Set)
**Use when:** connected components, cycle detection, Kruskal's MST

```
Initial:   A  B  C  D  E   (each node = own root)

union(A,B):    A         union(C,D): A    C
               |                    |    |
               B                    B    D

union(B,C):        A        find(D) with path compression:
                  /|\         D→C→A  becomes  D→A directly
                 B C D                         C→A directly
union(A,E):
                  A
                / | \ \
               B  C  D  E
```
```python
class UF:
    def __init__(self, n):
        self.p = list(range(n))
        self.rank = [0] * n
    def find(self, x):
        if self.p[x] != x: self.p[x] = self.find(self.p[x])
        return self.p[x]
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py: return False
        if self.rank[px] < self.rank[py]: px, py = py, px
        self.p[py] = px
        if self.rank[px] == self.rank[py]: self.rank[px] += 1
        return True
```

---

### Heap Patterns
```
Min-Heap structure (tree ↔ array):
        1              [_, 1, 3, 2, 7, 4, 5, 6]
       / \                  ↑  (1-indexed)
      3   2            parent(i) = i//2
     / \ / \           left(i)   = 2*i
    7  4 5  6          right(i)  = 2*i+1

K-largest: maintain min-heap of size k
  stream: [3,1,7,4,9,2]  k=3
  heap after each:
    [3] → [1,3] → [1,3,7] → [3,4,7] → [4,7,9] ← answer={4,7,9}
    (pop 1 when 4 arrives, pop 3 when 9 arrives)
```
```python
import heapq

def k_largest(nums, k):
    return heapq.nlargest(k, nums)   # O(n log k)

def merge_k(lists):
    pq, res = [], []
    for i, lst in enumerate(lists):
        if lst: heapq.heappush(pq, (lst[0], i, 0))
    while pq:
        val, i, j = heapq.heappop(pq)
        res.append(val)
        if j+1 < len(lists[i]):
            heapq.heappush(pq, (lists[i][j+1], i, j+1))
    return res
```

---

### Monotonic Stack
**Use when:** next greater/smaller element, largest rectangle, trapping rain

```
arr = [ 2,  1,  5,  3,  4]
nge = [ 5,  5, -1,  4, -1]

Stack trace (stores indices, keep decreasing values):
i=0: push 0          stack=[0]        (val=2)
i=1: 1<2, push 1     stack=[0,1]      (val=2,1)
i=2: 5>1→pop1 nge[1]=5
     5>2→pop0 nge[0]=5, push 2        stack=[2]
i=3: 3<5, push 3     stack=[2,3]      (val=5,3)
i=4: 4>3→pop3 nge[3]=4, push 4       stack=[2,4]
end: nge[2]=-1, nge[4]=-1

Trapping Rain Water (same idea, use max-stack):
  bar: [0,1,0,2,1,0,1,3,2,1,2,1]
       _
      _|_         _
  _  |   |_  _  |_|_  _
  |__|   | ||_||     ||_|
  water trapped = Σ min(maxL,maxR) - height
```
```python
def next_greater(arr):
    res, stack = [-1]*len(arr), []
    for i, v in enumerate(arr):
        while stack and arr[stack[-1]] < v:
            res[stack.pop()] = v
        stack.append(i)
    return res
```

---

### Waymo-Specific Focus Areas

**Autonomous Driving context — expect:**
- **Graph problems** (road networks, path planning, Dijkstra/A*)
- **Geometry** (point-in-polygon, line intersection, convex hull)
- **Interval/scheduling** (sensor fusion timing, trajectory overlap)
- **Simulation/state machines** (vehicle states, multi-agent)
- **Spatial data** (2D grids, flood fill, BFS on maps)

**Interview Tips:**
1. **Clarify first** — ask about constraints, edge cases, expected output format
2. **State complexity upfront** — time + space before coding
3. **Think aloud** — Waymo values reasoning about safety-critical systems
4. **Start brute force, optimize** — show the progression explicitly
5. **Test with edge cases** — empty input, single element, all same values
6. **For grid problems** — confirm 4-directional vs 8-directional movement

**Common Python Gotchas:**
```python
mid = (lo + hi) // 2               # no overflow in Python
from collections import defaultdict, Counter
d = defaultdict(list)
points.sort(key=lambda p: (p[0], p[1]))
heapq.heappush(pq, (priority, data))
float('inf'), float('-inf')
arr[l:r+1] = arr[l:r+1][::-1]     # in-place reverse slice
```

**Geometry Essentials:**
```python
def cross(O, A, B):   # > 0: CCW, < 0: CW, == 0: collinear
    return (A[0]-O[0])*(B[1]-O[1]) - (A[1]-O[1])*(B[0]-O[0])

import math
def dist(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def manhattan(p1, p2):
    return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])
```

---

*Key mindset: Waymo builds safety-critical systems — show precision, handle edge cases, and explain trade-offs clearly.*

---

## PAGE 3 — Advanced Data Structures & Algorithms (Staff Level)

---

### Trie (Prefix Tree)
**Use when:** autocomplete, prefix search, word dictionary, IP routing

```
insert("cat","car","card","care","bat"):

root
├── c
│   └── a
│       ├── t*           "cat"
│       └── r*           "car"
│           ├── d*       "card"
│           └── e*       "care"
└── b
    └── a
        └── t*           "bat"

search("car")  → root→c→a→r  is_end=True  ✓
starts_with("ca") → root→c→a  exists       ✓
search("ca")   → root→c→a  is_end=False   ✗
```
```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self): self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for c in word:
            node = node.children.setdefault(c, TrieNode())
        node.is_end = True

    def search(self, word):
        node = self.root
        for c in word:
            if c not in node.children: return False
            node = node.children[c]
        return node.is_end

    def starts_with(self, prefix):
        node = self.root
        for c in prefix:
            if c not in node.children: return False
            node = node.children[c]
        return True
```

---

### Segment Tree (Range Query + Point Update)
**Use when:** range sum/min/max queries with updates — O(log n) both ops

```
arr = [1, 3, 2, 7, 9, 11]   n=6, sum segment tree:

                  [0..5]=33
                /            \
         [0..2]=6            [3..5]=27
         /      \            /       \
     [0..1]=4  [2]=2    [3..4]=16  [5]=11
     /     \             /     \
  [0]=1  [1]=3        [3]=7   [4]=9

Internal array (1-indexed, size 2n):
  idx: 1   2   3   4   5   6   7   8   9   10  11
  val: 33  6   27  4   2   16  11  1   3   7   9
  (leaves start at index n=6)

Query [1..4]: walk up from leaves, sum partial nodes
Update idx 2: update leaf, propagate up to root — O(log n)
```
```python
class SegTree:
    def __init__(self, n):
        self.n = n
        self.tree = [0] * (2 * n)

    def update(self, i, val):
        i += self.n
        self.tree[i] = val
        while i > 1:
            i >>= 1
            self.tree[i] = self.tree[2*i] + self.tree[2*i+1]

    def query(self, l, r):  # [l, r) half-open
        res, l, r = 0, l + self.n, r + self.n
        while l < r:
            if l & 1: res += self.tree[l]; l += 1
            if r & 1: r -= 1; res += self.tree[r]
            l >>= 1; r >>= 1
        return res
```

---

### Fenwick Tree / BIT (Prefix Sums with Updates)
**Use when:** running prefix sums with point updates — simpler than SegTree

```
BIT[i] is responsible for sum of range ending at i,
of length = lowbit(i) = i & (-i):

idx:    1   2   3   4   5   6   7   8
binary: 001 010 011 100 101 110 111 1000
lowbit: 1   2   1   4   1   2   1   8
range: [1] [1-2][3] [1-4][5][5-6][7] [1-8]

BIT[4] covers [1..4]   (lowbit(4)=4)
BIT[6] covers [5..6]   (lowbit(6)=2)
BIT[8] covers [1..8]   (lowbit(8)=8)

Update(3, +5): touch idx 3 → 4 → 8 → ...
Query(6):      sum = BIT[6] + BIT[4]  (6→4→0, strip lowbit)
```
```python
class BIT:
    def __init__(self, n):
        self.n = n
        self.tree = [0] * (n + 1)

    def update(self, i, delta):  # 1-indexed
        while i <= self.n:
            self.tree[i] += delta
            i += i & (-i)

    def query(self, i):  # prefix sum [1..i]
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & (-i)
        return s

    def range_query(self, l, r):
        return self.query(r) - self.query(l - 1)
```

---

### Topological Sort (Kahn's BFS)
**Use when:** dependency ordering, course schedule, build systems, DAG processing

```
DAG:  A → B → D
      ↓   ↑
      C ——┘       (A→B, A→C, C→B, B→D)

indegree: {A:0, B:2, C:1, D:1}

Step 1: queue = [A]  (indegree 0)
Step 2: pop A → order=[A], decrement B→1, C→0 → queue=[C]
Step 3: pop C → order=[A,C], decrement B→0    → queue=[B]
Step 4: pop B → order=[A,C,B], decrement D→0  → queue=[D]
Step 5: pop D → order=[A,C,B,D] ✓

Cycle detection: if len(order) < n → cycle exists!
```
```python
from collections import deque

def topo_sort(n, edges):
    indegree = [0] * n
    graph = [[] for _ in range(n)]
    for u, v in edges:
        graph[u].append(v)
        indegree[v] += 1
    q = deque(i for i in range(n) if indegree[i] == 0)
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in graph[u]:
            indegree[v] -= 1
            if indegree[v] == 0: q.append(v)
    return order if len(order) == n else []  # [] = cycle
```

---

### A* Search (Heuristic Shortest Path)
**Use when:** grid pathfinding with known goal — beats Dijkstra when heuristic is tight

```
S=start, G=goal, #=wall     f = g + h
                             g = cost so far
  S . . # .                 h = manhattan to G (admissible)
  . # . . .
  . # . . .
  . . . . G

Dijkstra explores uniformly:     A* steers toward G:
  ○○○○○                            ○
  ○○○○○                           ○○○
  ○○○○○                          ○○○○○
  ○○○○G                         ○○○○○G

Nodes visited: ~all           Nodes visited: ~corridor
```
```python
import heapq

def astar(grid, start, goal):
    def h(p): return abs(p[0]-goal[0]) + abs(p[1]-goal[1])
    pq = [(h(start), 0, start)]   # (f=g+h, g, node)
    g = {start: 0}
    while pq:
        f, cost, (r, c) = heapq.heappop(pq)
        if (r, c) == goal: return cost
        if cost > g.get((r,c), float('inf')): continue
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0<=nr<len(grid) and 0<=nc<len(grid[0]) and grid[nr][nc] != '#':
                ng = cost + 1
                if ng < g.get((nr,nc), float('inf')):
                    g[(nr,nc)] = ng
                    heapq.heappush(pq, (ng + h((nr,nc)), ng, (nr,nc)))
    return -1
```

---

### Strongly Connected Components (Kosaraju's)
**Use when:** SCCs in directed graph, 2-SAT, condensation DAG

```
Graph G:                  Reverse G^T:
  A → B                     A ← B
  ↑   ↓          →          ↑   ↓
  C ← D                     C → D
        \                          \
         E                          E

Step 1: DFS on G, record finish order: B, A, D, C, E
         (last to finish = highest in order)
Step 2: DFS on G^T in reverse finish order:
  Start E: SCC={E}
  Start C: C→A→B→D  SCC={A,B,C,D}  ← condensation root

Condensation DAG: {A,B,C,D} → {E}
```
```python
def kosaraju(n, edges):
    graph, rgraph = [[] for _ in range(n)], [[] for _ in range(n)]
    for u, v in edges:
        graph[u].append(v); rgraph[v].append(u)
    visited, order = set(), []
    def dfs1(u):
        visited.add(u)
        for v in graph[u]:
            if v not in visited: dfs1(v)
        order.append(u)
    def dfs2(u, comp):
        visited.add(u); comp.append(u)
        for v in rgraph[u]:
            if v not in visited: dfs2(v, comp)
    for i in range(n):
        if i not in visited: dfs1(i)
    visited.clear()
    sccs = []
    for u in reversed(order):
        if u not in visited:
            comp = []; dfs2(u, comp); sccs.append(comp)
    return sccs
```

---

### Sweep Line (Intervals / Events)
**Use when:** meeting rooms, skyline, rectangle area, event scheduling

```
Meeting rooms II:
  ────────────────────────────────→ time
  [=======A=======]
           [====B====]   ← A and B overlap → need 2 rooms
                   [==C==]
                          [=D=]

Events (time, delta):  sort, apply running sum
  (sA,+1)(sB,+1)(eA,-1)(sC,+1)(eB,-1)(eC,-1)(sD,+1)(eD,-1)
   1      1      -1     1      -1     -1      1     -1
running:  1  2    1     2       1      0      1      0
max = 2 ← answer

Skyline: use max-heap of (−height, end) for active buildings
```
```python
def min_rooms(intervals):
    events = []
    for s, e in intervals:
        events.append((s, 1))
        events.append((e, -1))
    events.sort()
    rooms = cur = 0
    for _, delta in events:
        cur += delta
        rooms = max(rooms, cur)
    return rooms
```

---

### Bellman-Ford & Convex Hull

```
Bellman-Ford — negative weights:
  Pass 1: relax all edges → settle 1-hop paths
  Pass 2: relax all edges → settle 2-hop paths
  ...
  Pass n-1: all shortest paths settled
  Pass n:   if any relaxation occurs → negative cycle!

Convex Hull (Graham Scan):
  Points:              Hull (CCW):
    *  *                 *———*
  *  *  *  *           *       *
    *  *    *    →       *     *
  *    *  *               *———*

  1. Find bottom-left anchor
  2. Sort by polar angle from anchor
  3. Walk points: if right turn (CW cross ≤ 0) → pop stack
  4. All remaining = hull vertices
```
```python
def bellman_ford(n, edges, src):
    dist = [float('inf')] * n
    dist[src] = 0
    for _ in range(n - 1):
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
    for u, v, w in edges:
        if dist[u] + w < dist[v]: return None  # negative cycle
    return dist

def convex_hull(points):
    points = sorted(set(points))
    def cross(O, A, B):
        return (A[0]-O[0])*(B[1]-O[1]) - (A[1]-O[1])*(B[0]-O[0])
    lower, upper = [], []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0: lower.pop()
        lower.append(p)
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0: upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]
```

---

## PAGE 4 — System Design, Advanced DP & Staff Expectations

---

### Staff-Level Problem-Solving Framework

**When given a hard problem, structure your response:**
1. **Restate** — confirm understanding, clarify ambiguities
2. **Brute force** — state it, give complexity, don't code it
3. **Optimize** — identify bottleneck (time? space?), propose better approach
4. **Implement** — clean, modular code with meaningful names
5. **Verify** — trace through 2-3 examples including edge cases
6. **Complexity analysis** — precise, with justification
7. **Trade-offs** — discuss what you'd change at scale / in production

**Staff differentiators vs senior:**
- You should be able to **derive** the algorithm, not just recall it
- Proactively discuss **correctness proofs** (loop invariants, induction)
- Note **constant factor** differences (cache locality, branch prediction)
- Suggest **alternative implementations** and explain when each is better
- Connect algorithm choices to **real-world system constraints**

---

### Advanced DP Patterns

```
Interval DP — build answers for larger intervals from smaller:
  dp[i][j] = best for subarray [i..j]

  length=1: ■ ■ ■ ■     (base cases, diagonal)
  length=2: ■ ■ ■       (use length-1 results)
  length=3: ■ ■         (use length-2 results)
  length=4: ■           (answer at dp[0][n-1])

Bitmask DP — TSP (n cities, 2^n states):
  mask = visited set as bitmask
  dp[mask][u] = min cost visiting cities in mask, ending at u

  mask=0001 (only city 0):  dp[1][0]=0
  mask=0011 (cities 0,1):   dp[3][1]=dist[0][1]
  mask=0111 (cities 0,1,2): dp[7][2]=min over entry points
  mask=1111 (all):          + return to 0 = answer

Tree DP — rerooting technique:
      1
     / \               down[v] = best answer rooted at v
    2   3              up[v]   = best answer from parent side
   / \                 full[v] = max(down[v], up[v])
  4   5
  Process: 1 DFS down (children → parent)
           1 DFS up   (parent  → children, reroot)
```
```python
def interval_dp(arr):
    n = len(arr)
    dp = [[0]*n for _ in range(n)]
    for length in range(2, n+1):
        for i in range(n - length + 1):
            j = i + length - 1
            for k in range(i, j):   # split point
                dp[i][j] = max(dp[i][j], dp[i][k] + dp[k+1][j] + cost(i,k,j))
    return dp[0][n-1]

def tsp(dist):
    n = len(dist)
    INF = float('inf')
    dp = [[INF]*n for _ in range(1<<n)]
    dp[1][0] = 0
    for mask in range(1<<n):
        for u in range(n):
            if dp[mask][u] == INF or not (mask >> u & 1): continue
            for v in range(n):
                if mask >> v & 1: continue
                nmask = mask | (1 << v)
                dp[nmask][v] = min(dp[nmask][v], dp[mask][u] + dist[u][v])
    return min(dp[(1<<n)-1][i] + dist[i][0] for i in range(1,n))

def tree_dp(root, parent, graph, values):
    include = values[root]
    exclude = 0
    for child in graph[root]:
        if child == parent: continue
        ci, ce = tree_dp(child, root, graph, values)
        include += ce
        exclude += max(ci, ce)
    return include, exclude
```

---

### Concurrency Patterns
**Waymo: real-time multi-sensor pipelines — concurrency is non-trivial**

```
Producer-Consumer (bounded buffer):

  Producer                    Consumer
  ────────                    ────────
  put(item) ──→ [■ ■ ■ □ □] ──→ get() → process
               queue (maxsize)
               blocks if full    blocks if empty

Deadlock (avoid with lock ordering):
  Thread 1: lock(A) then lock(B)
  Thread 2: lock(B) then lock(A)  ← circular wait!
  Fix: always acquire locks in same global order

Race condition:
  x = 0
  T1: x = x + 1   ← read(0), write(1)
  T2: x = x + 1   ← read(0), write(1)  ← lost update!
  Result: x=1 instead of 2  → use Lock or atomic ops
```
```python
import threading, queue

buf = queue.Queue(maxsize=10)

def producer(data):
    for item in data:
        buf.put(item)   # blocks if full
    buf.put(None)       # sentinel

def consumer():
    while True:
        item = buf.get()
        if item is None: break
        process(item)

class RWLock:
    def __init__(self):
        self._readers = 0
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
    def acquire_read(self):
        with self._lock:
            self._readers += 1
            if self._readers == 1: self._write_lock.acquire()
    def release_read(self):
        with self._lock:
            self._readers -= 1
            if self._readers == 0: self._write_lock.release()
    def acquire_write(self): self._write_lock.acquire()
    def release_write(self): self._write_lock.release()
```

---

### System Design at Scale — Autonomous Driving Context

```
Sensor data pipeline (LiDAR+camera+radar at ~100Hz):

  [LiDAR]──┐
  [Camera]─┼──→ [Kafka topics] ──→ [Stream processor] ──→ [Object store]
  [Radar]──┘    (per sensor)        (fusion, detection)    (GCS/S3)
                                           │
                                    [Time-series DB]  ←── monitoring
                                    [HD Map cache]    ←── tile serving

Scale numbers to cite:
  ~10 sensors × 100Hz × ~1MB/frame = ~1GB/s per vehicle
  Fleet of 1000 vehicles = ~1TB/s ingestion
  → must shard by vehicle_id, use columnar storage (Parquet)
```

| Component | When to use | Key trade-offs |
|-----------|------------|----------------|
| Kafka | Decoupled sensor streams | Throughput vs latency |
| InfluxDB | Sensor telemetry | Write-optimized, hard to join |
| GCS/S3 | Raw logs, sensor data | Cheap, high retrieval latency |
| Redis | Map tiles, frequent lookups | Volatile, consistency risk |
| Spanner | Fleet state, strong consistency | Expensive, global latency |

**System design answer structure (45-min):**
1. Clarify requirements (5 min) — scale, consistency, latency targets
2. High-level architecture (10 min) — components, data flow, APIs
3. Deep dive on hardest component (15 min) — data model, bottlenecks
4. Failure modes & reliability (10 min) — what fails, detect, recover
5. Scaling & trade-offs (5 min) — hotspots, sharding, observability

---

### Complexity & Optimization Signals

**Interviewer probes → your response:**
```
"Can you do better?"        → Is there a lower bound? O(n log n) for comparison sorts is proven
"What if n = 10^9?"         → O(n) too slow; need O(log n)/O(1); look for math/formula
"Streaming data?"           → Online algorithm, sliding window, count-min sketch for approx
"What about memory?"        → Roll DP array, generators, external sort, O(1) space tricks
"Production concerns?"      → Overflow, float precision, thread safety, retries, observability
"What if graph has cycles?"  → Detect with DFS color (white/gray/black) or Union-Find
```

**Amortized analysis — cite these cold:**
```
Dynamic array append:   O(1) amortized  (doubling → total work = 1+2+4+...+n = 2n)
Union-Find (pc+rank):   O(α(n)) ≈ O(1)  (inverse Ackermann)
Monotonic stack sweep:  O(n) total      (each element pushed/popped exactly once)
Splay tree ops:         O(log n) amortized
```

---

### Staff Behavioral Signals — Technical Leadership

| Question type | What Waymo is probing |
|--------------|----------------------|
| "System you designed end-to-end" | Scope, ambiguity handling, stakeholder alignment |
| "Disagreement with a senior engineer" | Influence without authority, data-driven |
| "Production incident you led" | Incident command, RCA, blameless culture |
| "When to take on tech debt?" | Engineering judgment vs business context |
| "Hard architectural trade-off" | First principles, reversibility awareness |

**Staff-level signal phrases:**
- *"I set the technical direction and delegated implementation to..."*
- *"I wrote the design doc and got buy-in from X, Y, Z teams..."*
- *"My runbook reduced MTTR from X hours to Y minutes..."*
- *"I made the call to deprecate X — maintenance burden outweighed benefits..."*
- *"I mentored two engineers through this — here's how I structured it..."*

---

### Quick Complexity Cheat Card

```
Sorting:           O(n log n)  — comparison lower bound proven
Binary search:     O(log n)
BFS/DFS:           O(V + E)
Dijkstra (heap):   O((V + E) log V)
Bellman-Ford:      O(V · E)
Floyd-Warshall:    O(V³)       — all-pairs shortest path
Prim's MST:        O(E log V)
Kruskal's MST:     O(E log E)
Topo sort:         O(V + E)
Segment tree:      O(log n) query/update,  O(n) build
Fenwick tree:      O(log n) query/update,  O(n) build
Trie:              O(L) insert/search      L = word length
Convex hull:       O(n log n)
TSP (bitmask DP):  O(2^n · n²)
Interval DP:       O(n³)
A* (best case):    O(E)  with perfect heuristic
```

---

*Staff mindset: You are evaluated not just on solving problems, but on how you **frame**, **decompose**, and **communicate** them. Show that you'd make the same choices in production code with millions of users — and lives — at stake.*
