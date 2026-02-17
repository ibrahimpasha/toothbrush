
---

## Table of Contents
1. [Arrays & Two Pointers](#arrays--two-pointers) (Problems 1-10)
2. [Strings](#strings) (Problems 11-17)
3. [Linked Lists](#linked-lists) (Problems 18-23)
4. [Trees & Graphs](#trees--graphs) (Problems 24-32)
5. [Dynamic Programming](#dynamic-programming) (Problems 33-40)
6. [Binary Search](#binary-search) (Problems 41-44)
7. [Stack & Queue](#stack--queue) (Problems 45-47)
8. [Math & Bit Manipulation](#math--bit-manipulation) (Problems 48-50)

---

## Arrays & Two Pointers

### Problem 1: Two Sum
**Difficulty:** Easy | **LC #1**

Given an array of integers `nums` and an integer `target`, return indices of the two numbers that add up to target.

```python
def twoSum(nums: list[int], target: int) -> list[int]:
    seen = {}  # value -> index
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []

# Time: O(n), Space: O(n)

# Examples:
# Input: nums = [2,7,11,15], target = 9
# Output: [0,1]  (nums[0] + nums[1] = 2 + 7 = 9)

# Input: nums = [3,2,4], target = 6
# Output: [1,2]  (nums[1] + nums[2] = 2 + 4 = 6)
```

---

### Problem 2: Three Sum
**Difficulty:** Medium | **LC #15**

Find all unique triplets in the array that sum to zero.

```python
def threeSum(nums: list[int]) -> list[list[int]]:
    nums.sort()
    result = []
    
    for i in range(len(nums) - 2):
        # Skip duplicates for first element
        if i > 0 and nums[i] == nums[i-1]:
            continue
        
        left, right = i + 1, len(nums) - 1
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total < 0:
                left += 1
            elif total > 0:
                right -= 1
            else:
                result.append([nums[i], nums[left], nums[right]])
                # Skip duplicates
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1
    
    return result

# Time: O(n²), Space: O(1) excluding output

# Examples:
# Input: nums = [-1,0,1,2,-1,-4]
# Output: [[-1,-1,2],[-1,0,1]]

# Input: nums = [0,0,0,0]
# Output: [[0,0,0]]

# Input: nums = [1,2,-2,-1]
# Output: []
```

---

### Problem 3: Container With Most Water
**Difficulty:** Medium | **LC #11**

Find two lines that together with the x-axis form a container that holds the most water.

```python
def maxArea(height: list[int]) -> int:
    left, right = 0, len(height) - 1
    max_water = 0
    
    while left < right:
        width = right - left
        h = min(height[left], height[right])
        max_water = max(max_water, width * h)
        
        # Move the shorter line inward
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    
    return max_water

# Time: O(n), Space: O(1)

# Examples:
# Input: height = [1,8,6,2,5,4,8,3,7]
# Output: 49  (between index 1 and 8, width=7, height=min(8,7)=7, area=49)

# Input: height = [1,1]
# Output: 1
```

---

### Problem 4: Product of Array Except Self
**Difficulty:** Medium | **LC #238**

Return an array where each element is the product of all elements except itself, without using division.

```python
def productExceptSelf(nums: list[int]) -> list[int]:
    n = len(nums)
    result = [1] * n
    
    # Left pass: result[i] = product of all elements to the left
    left_product = 1
    for i in range(n):
        result[i] = left_product
        left_product *= nums[i]
    
    # Right pass: multiply by product of all elements to the right
    right_product = 1
    for i in range(n - 1, -1, -1):
        result[i] *= right_product
        right_product *= nums[i]
    
    return result

# Time: O(n), Space: O(1) excluding output

# Examples:
# Input: nums = [1,2,3,4]
# Output: [24,12,8,6]  (24=2*3*4, 12=1*3*4, 8=1*2*4, 6=1*2*3)

# Input: nums = [-1,1,0,-3,3]
# Output: [0,0,9,0,0]
```

---

### Problem 5: Maximum Subarray (Kadane's Algorithm)
**Difficulty:** Medium | **LC #53**

Find the contiguous subarray with the largest sum.

```python
def maxSubArray(nums: list[int]) -> int:
    max_sum = nums[0]
    current_sum = nums[0]
    
    for i in range(1, len(nums)):
        # Either extend current subarray or start new one
        current_sum = max(nums[i], current_sum + nums[i])
        max_sum = max(max_sum, current_sum)
    
    return max_sum

# Time: O(n), Space: O(1)

# Examples:
# Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
# Output: 6  (subarray [4,-1,2,1] has sum 6)

# Input: nums = [5,4,-1,7,8]
# Output: 23  (entire array)

# Input: nums = [-1]
# Output: -1
```

---

### Problem 6: Merge Intervals
**Difficulty:** Medium | **LC #56**

Merge all overlapping intervals.

```python
def merge(intervals: list[list[int]]) -> list[list[int]]:
    if not intervals:
        return []
    
    # Sort by start time
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    
    for start, end in intervals[1:]:
        # If overlapping with last merged interval
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    
    return merged

# Time: O(n log n), Space: O(n)

# Examples:
# Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
# Output: [[1,6],[8,10],[15,18]]  ([1,3] and [2,6] merge to [1,6])

# Input: intervals = [[1,4],[4,5]]
# Output: [[1,5]]  (touching intervals merge)
```

---

### Problem 7: Find Peak Element
**Difficulty:** Medium | **LC #162**

Find a peak element (greater than neighbors) in O(log n) time.

```python
def findPeakElement(nums: list[int]) -> int:
    left, right = 0, len(nums) - 1
    
    while left < right:
        mid = (left + right) // 2
        if nums[mid] > nums[mid + 1]:
            # Peak is on the left side (including mid)
            right = mid
        else:
            # Peak is on the right side
            left = mid + 1
    
    return left

# Time: O(log n), Space: O(1)

# Examples:
# Input: nums = [1,2,3,1]
# Output: 2  (index of peak element 3)

# Input: nums = [1,2,1,3,5,6,4]
# Output: 5  (index of peak element 6, or 1 for peak 2)
```

---

### Problem 8: Rotate Array
**Difficulty:** Medium | **LC #189**

Rotate array to the right by k steps.

```python
def rotate(nums: list[int], k: int) -> None:
    n = len(nums)
    k = k % n  # Handle k > n
    
    def reverse(start, end):
        while start < end:
            nums[start], nums[end] = nums[end], nums[start]
            start += 1
            end -= 1
    
    # Reverse entire array, then reverse first k, then reverse rest
    reverse(0, n - 1)
    reverse(0, k - 1)
    reverse(k, n - 1)

# Time: O(n), Space: O(1)

# Examples:
# Input: nums = [1,2,3,4,5,6,7], k = 3
# Output: [5,6,7,1,2,3,4]

# Input: nums = [-1,-100,3,99], k = 2
# Output: [3,99,-1,-100]
```

---

### Problem 9: Move Zeroes
**Difficulty:** Easy | **LC #283**

Move all zeroes to the end while maintaining relative order of non-zero elements.

```python
def moveZeroes(nums: list[int]) -> None:
    insert_pos = 0
    
    # Move all non-zero elements to the front
    for i in range(len(nums)):
        if nums[i] != 0:
            nums[insert_pos] = nums[i]
            insert_pos += 1
    
    # Fill remaining positions with zeros
    while insert_pos < len(nums):
        nums[insert_pos] = 0
        insert_pos += 1

# Time: O(n), Space: O(1)

# Examples:
# Input: nums = [0,1,0,3,12]
# Output: [1,3,12,0,0]

# Input: nums = [0,0,1]
# Output: [1,0,0]
```

---

### Problem 10: Trapping Rain Water
**Difficulty:** Hard | **LC #42**

Calculate how much water can be trapped after raining.

```python
def trap(height: list[int]) -> int:
    if not height:
        return 0
    
    left, right = 0, len(height) - 1
    left_max, right_max = height[left], height[right]
    water = 0
    
    while left < right:
        if left_max < right_max:
            left += 1
            left_max = max(left_max, height[left])
            water += left_max - height[left]
        else:
            right -= 1
            right_max = max(right_max, height[right])
            water += right_max - height[right]
    
    return water

# Time: O(n), Space: O(1)

# Examples:
# Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
# Output: 6  (water trapped in valleys)

# Input: height = [4,2,0,3,2,5]
# Output: 9
```

---

## Strings

### Problem 11: Valid Palindrome
**Difficulty:** Easy | **LC #125**

Check if a string is a palindrome, considering only alphanumeric characters.

```python
def isPalindrome(s: str) -> bool:
    left, right = 0, len(s) - 1
    
    while left < right:
        # Skip non-alphanumeric
        while left < right and not s[left].isalnum():
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1
        
        if s[left].lower() != s[right].lower():
            return False
        
        left += 1
        right -= 1
    
    return True

# Time: O(n), Space: O(1)

# Examples:
# Input: s = "A man, a plan, a canal: Panama"
# Output: True  (reads "amanaplanacanalpanama" both ways)

# Input: s = "race a car"
# Output: False

# Input: s = " "
# Output: True  (empty after removing non-alphanumeric)
```

---

### Problem 12: Longest Substring Without Repeating Characters
**Difficulty:** Medium | **LC #3**

Find the length of the longest substring without repeating characters.

```python
def lengthOfLongestSubstring(s: str) -> int:
    char_index = {}  # char -> last seen index
    max_length = 0
    start = 0
    
    for i, char in enumerate(s):
        # If char seen and within current window
        if char in char_index and char_index[char] >= start:
            start = char_index[char] + 1
        
        char_index[char] = i
        max_length = max(max_length, i - start + 1)
    
    return max_length

# Time: O(n), Space: O(min(n, alphabet_size))

# Examples:
# Input: s = "abcabcbb"
# Output: 3  (substring "abc")

# Input: s = "bbbbb"
# Output: 1  (substring "b")

# Input: s = "pwwkew"
# Output: 3  (substring "wke")
```

---

### Problem 13: Longest Palindromic Substring
**Difficulty:** Medium | **LC #5**

Find the longest palindromic substring.

```python
def longestPalindrome(s: str) -> str:
    if not s:
        return ""
    
    def expand_around_center(left: int, right: int) -> str:
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return s[left + 1:right]
    
    result = ""
    for i in range(len(s)):
        # Odd length palindrome
        odd = expand_around_center(i, i)
        # Even length palindrome
        even = expand_around_center(i, i + 1)
        
        result = max(result, odd, even, key=len)
    
    return result

# Time: O(n²), Space: O(1)

# Examples:
# Input: s = "babad"
# Output: "bab"  (or "aba", both valid)

# Input: s = "cbbd"
# Output: "bb"
```

---

### Problem 14: Group Anagrams
**Difficulty:** Medium | **LC #49**

Group strings that are anagrams of each other.

```python
from collections import defaultdict

def groupAnagrams(strs: list[str]) -> list[list[str]]:
    groups = defaultdict(list)
    
    for s in strs:
        # Use sorted string as key
        key = tuple(sorted(s))
        groups[key].append(s)
    
    return list(groups.values())

# Alternative: Use character count as key (faster for long strings)
def groupAnagrams_v2(strs: list[str]) -> list[list[str]]:
    groups = defaultdict(list)
    
    for s in strs:
        count = [0] * 26
        for c in s:
            count[ord(c) - ord('a')] += 1
        groups[tuple(count)].append(s)
    
    return list(groups.values())

# Time: O(n * k log k) or O(n * k), Space: O(n * k)

# Examples:
# Input: strs = ["eat","tea","tan","ate","nat","bat"]
# Output: [["eat","tea","ate"],["tan","nat"],["bat"]]

# Input: strs = [""]
# Output: [[""]]

# Input: strs = ["a"]
# Output: [["a"]]
```

---

### Problem 15: Valid Parentheses
**Difficulty:** Easy | **LC #20**

Check if the input string has valid bracket matching.

```python
def isValid(s: str) -> bool:
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    
    for char in s:
        if char in mapping:
            # Closing bracket
            if not stack or stack.pop() != mapping[char]:
                return False
        else:
            # Opening bracket
            stack.append(char)
    
    return len(stack) == 0

# Time: O(n), Space: O(n)

# Examples:
# Input: s = "()[]{}"
# Output: True

# Input: s = "(]"
# Output: False

# Input: s = "([)]"
# Output: False

# Input: s = "{[]}"
# Output: True
```

---

### Problem 16: Minimum Window Substring
**Difficulty:** Hard | **LC #76**

Find the minimum window in s that contains all characters of t.

```python
from collections import Counter

def minWindow(s: str, t: str) -> str:
    if not s or not t:
        return ""
    
    t_count = Counter(t)
    required = len(t_count)
    
    left = 0
    formed = 0
    window_counts = {}
    
    result = (float('inf'), None, None)  # (length, left, right)
    
    for right, char in enumerate(s):
        window_counts[char] = window_counts.get(char, 0) + 1
        
        if char in t_count and window_counts[char] == t_count[char]:
            formed += 1
        
        # Contract window
        while formed == required:
            if right - left + 1 < result[0]:
                result = (right - left + 1, left, right)
            
            left_char = s[left]
            window_counts[left_char] -= 1
            if left_char in t_count and window_counts[left_char] < t_count[left_char]:
                formed -= 1
            left += 1
    
    return "" if result[0] == float('inf') else s[result[1]:result[2] + 1]

# Time: O(|s| + |t|), Space: O(|s| + |t|)

# Examples:
# Input: s = "ADOBECODEBANC", t = "ABC"
# Output: "BANC"  (smallest window containing A, B, C)

# Input: s = "a", t = "a"
# Output: "a"

# Input: s = "a", t = "aa"
# Output: ""  (not enough 'a's)
```

---

### Problem 17: String to Integer (atoi)
**Difficulty:** Medium | **LC #8**

Implement atoi to convert a string to an integer.

```python
def myAtoi(s: str) -> int:
    INT_MAX, INT_MIN = 2**31 - 1, -2**31
    
    s = s.lstrip()  # Remove leading whitespace
    if not s:
        return 0
    
    sign = 1
    i = 0
    
    # Handle sign
    if s[0] == '-':
        sign = -1
        i = 1
    elif s[0] == '+':
        i = 1
    
    result = 0
    while i < len(s) and s[i].isdigit():
        digit = int(s[i])
        
        # Check overflow before adding
        if result > (INT_MAX - digit) // 10:
            return INT_MAX if sign == 1 else INT_MIN
        
        result = result * 10 + digit
        i += 1
    
    return sign * result

# Time: O(n), Space: O(1)

# Examples:
# Input: s = "   -42"
# Output: -42

# Input: s = "4193 with words"
# Output: 4193  (stops at non-digit)

# Input: s = "words and 987"
# Output: 0  (no leading digits)

# Input: s = "-91283472332"
# Output: -2147483648  (clamped to INT_MIN)
```

---

## Linked Lists

### Problem 18: Reverse Linked List
**Difficulty:** Easy | **LC #206**

Reverse a singly linked list.

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverseList(head: ListNode) -> ListNode:
    prev = None
    current = head
    
    while current:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node
    
    return prev

# Recursive version
def reverseList_recursive(head: ListNode) -> ListNode:
    if not head or not head.next:
        return head
    
    new_head = reverseList_recursive(head.next)
    head.next.next = head
    head.next = None
    
    return new_head

# Time: O(n), Space: O(1) iterative, O(n) recursive

# Examples:
# Input: head = [1,2,3,4,5]
# Output: [5,4,3,2,1]

# Input: head = [1,2]
# Output: [2,1]

# Input: head = []
# Output: []
```

---

### Problem 19: Merge Two Sorted Lists
**Difficulty:** Easy | **LC #21**

Merge two sorted linked lists into one sorted list.

```python
def mergeTwoLists(l1: ListNode, l2: ListNode) -> ListNode:
    dummy = ListNode(0)
    current = dummy
    
    while l1 and l2:
        if l1.val <= l2.val:
            current.next = l1
            l1 = l1.next
        else:
            current.next = l2
            l2 = l2.next
        current = current.next
    
    current.next = l1 or l2
    return dummy.next

# Time: O(n + m), Space: O(1)

# Examples:
# Input: l1 = [1,2,4], l2 = [1,3,4]
# Output: [1,1,2,3,4,4]

# Input: l1 = [], l2 = [0]
# Output: [0]
```

---

### Problem 20: Linked List Cycle
**Difficulty:** Easy | **LC #141**

Detect if a linked list has a cycle.

```python
def hasCycle(head: ListNode) -> bool:
    slow = fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    
    return False

# Time: O(n), Space: O(1)

# Examples:
# Input: head = [3,2,0,-4], pos = 1 (tail connects to index 1)
# Output: True

# Input: head = [1,2], pos = 0 (tail connects to index 0)
# Output: True

# Input: head = [1], pos = -1 (no cycle)
# Output: False
```

---

### Problem 21: Linked List Cycle II - Find Start
**Difficulty:** Medium | **LC #142**

Find the node where the cycle begins.

```python
def detectCycle(head: ListNode) -> ListNode:
    slow = fast = head
    
    # Find meeting point
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            break
    else:
        return None  # No cycle
    
    # Find cycle start
    slow = head
    while slow != fast:
        slow = slow.next
        fast = fast.next
    
    return slow

# Time: O(n), Space: O(1)

# Examples:
# Input: head = [3,2,0,-4], pos = 1
# Output: Node with value 2 (cycle starts at index 1)

# Input: head = [1], pos = -1
# Output: None (no cycle)
```

---

### Problem 22: Copy List with Random Pointer
**Difficulty:** Medium | **LC #138**

Deep copy a linked list where each node has a random pointer.

```python
class Node:
    def __init__(self, val, next=None, random=None):
        self.val = val
        self.next = next
        self.random = random

def copyRandomList(head: Node) -> Node:
    if not head:
        return None
    
    # Step 1: Create interleaved list (A -> A' -> B -> B' -> ...)
    current = head
    while current:
        copy = Node(current.val, current.next)
        current.next = copy
        current = copy.next
    
    # Step 2: Set random pointers for copies
    current = head
    while current:
        if current.random:
            current.next.random = current.random.next
        current = current.next.next
    
    # Step 3: Separate the two lists
    dummy = Node(0)
    copy_current = dummy
    current = head
    
    while current:
        copy_current.next = current.next
        current.next = current.next.next
        current = current.next
        copy_current = copy_current.next
    
    return dummy.next

# Time: O(n), Space: O(1) excluding output

# Examples:
# Input: head = [[7,null],[13,0],[11,4],[10,2],[1,0]]
#        (format: [val, random_index])
# Output: Deep copy with same structure

# Input: head = [[1,1],[2,1]]
# Output: [[1,1],[2,1]] (both nodes' random points to node at index 1)
```

---

### Problem 23: LRU Cache
**Difficulty:** Medium | **LC #146**

Design a Least Recently Used (LRU) cache.

```python
class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}  # key -> node
        
        # Doubly linked list with dummy head and tail
        self.head = DLLNode(0, 0)
        self.tail = DLLNode(0, 0)
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def _remove(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev
    
    def _add_to_front(self, node):
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node
    
    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        
        node = self.cache[key]
        self._remove(node)
        self._add_to_front(node)
        return node.val
    
    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self._remove(self.cache[key])
        
        node = DLLNode(key, value)
        self.cache[key] = node
        self._add_to_front(node)
        
        if len(self.cache) > self.capacity:
            # Remove LRU (node before tail)
            lru = self.tail.prev
            self._remove(lru)
            del self.cache[lru.key]

class DLLNode:
    def __init__(self, key, val):
        self.key = key
        self.val = val
        self.prev = None
        self.next = None

# Time: O(1) for both get and put

# Example:
# cache = LRUCache(2)
# cache.put(1, 1)        # cache: {1=1}
# cache.put(2, 2)        # cache: {1=1, 2=2}
# cache.get(1)           # returns 1, cache: {2=2, 1=1}
# cache.put(3, 3)        # evicts key 2, cache: {1=1, 3=3}
# cache.get(2)           # returns -1 (not found)
# cache.put(4, 4)        # evicts key 1, cache: {3=3, 4=4}
# cache.get(1)           # returns -1
# cache.get(3)           # returns 3
# cache.get(4)           # returns 4
```

---

## Trees & Graphs


### Problem 24: Maximum Depth of Binary Tree
**Difficulty:** Easy | **LC #104**

Find the maximum depth of a binary tree.

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def maxDepth(root: TreeNode) -> int:
    if not root:
        return 0
    return 1 + max(maxDepth(root.left), maxDepth(root.right))

# Iterative BFS
from collections import deque

def maxDepth_bfs(root: TreeNode) -> int:
    if not root:
        return 0
    
    queue = deque([root])
    depth = 0
    
    while queue:
        depth += 1
        for _ in range(len(queue)):
            node = queue.popleft()
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    
    return depth

# Time: O(n), Space: O(h) recursive, O(n) BFS

# Examples:
# Input: root = [3,9,20,null,null,15,7]
#            3
#           / \
#          9  20
#            /  \
#           15   7
# Output: 3

# Input: root = [1,null,2]
# Output: 2
```

---

### Problem 25: Validate Binary Search Tree
**Difficulty:** Medium | **LC #98**

Check if a binary tree is a valid BST.

```python
def isValidBST(root: TreeNode) -> bool:
    def validate(node, min_val, max_val):
        if not node:
            return True
        
        if node.val <= min_val or node.val >= max_val:
            return False
        
        return (validate(node.left, min_val, node.val) and
                validate(node.right, node.val, max_val))
    
    return validate(root, float('-inf'), float('inf'))

# Inorder traversal approach (BST inorder is sorted)
def isValidBST_inorder(root: TreeNode) -> bool:
    prev = float('-inf')
    
    def inorder(node):
        nonlocal prev
        if not node:
            return True
        
        if not inorder(node.left):
            return False
        
        if node.val <= prev:
            return False
        prev = node.val
        
        return inorder(node.right)
    
    return inorder(root)

# Time: O(n), Space: O(h)

# Examples:
# Input: root = [2,1,3]
#        2
#       / \
#      1   3
# Output: True

# Input: root = [5,1,4,null,null,3,6]
#        5
#       / \
#      1   4
#         / \
#        3   6
# Output: False (4's left child 3 < 4 but 3 < 5, violates BST)
```

---

### Problem 26: Lowest Common Ancestor of BST
**Difficulty:** Medium | **LC #235**

Find the lowest common ancestor of two nodes in a BST.

```python
def lowestCommonAncestor(root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
    while root:
        if p.val < root.val and q.val < root.val:
            root = root.left
        elif p.val > root.val and q.val > root.val:
            root = root.right
        else:
            return root
    return None

# Time: O(h), Space: O(1)

# Examples:
# Input: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 8
#            6
#          /   \
#         2     8
#        / \   / \
#       0   4 7   9
#          / \
#         3   5
# Output: 6 (LCA of 2 and 8)

# Input: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 4
# Output: 2 (node can be ancestor of itself)
```

---

### Problem 27: Binary Tree Level Order Traversal
**Difficulty:** Medium | **LC #102**

Return level order traversal of a binary tree.

```python
from collections import deque

def levelOrder(root: TreeNode) -> list[list[int]]:
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level)
    
    return result

# Time: O(n), Space: O(n)

# Examples:
# Input: root = [3,9,20,null,null,15,7]
# Output: [[3],[9,20],[15,7]]

# Input: root = [1]
# Output: [[1]]

# Input: root = []
# Output: []
```

---

### Problem 28: Serialize and Deserialize Binary Tree
**Difficulty:** Hard | **LC #297**

Design an algorithm to serialize and deserialize a binary tree.

```python
class Codec:
    def serialize(self, root: TreeNode) -> str:
        """Encodes a tree to a single string using preorder."""
        result = []
        
        def preorder(node):
            if not node:
                result.append("null")
                return
            result.append(str(node.val))
            preorder(node.left)
            preorder(node.right)
        
        preorder(root)
        return ",".join(result)
    
    def deserialize(self, data: str) -> TreeNode:
        """Decodes your encoded data to tree."""
        values = iter(data.split(","))
        
        def build():
            val = next(values)
            if val == "null":
                return None
            node = TreeNode(int(val))
            node.left = build()
            node.right = build()
            return node
        
        return build()

# Time: O(n), Space: O(n)

# Examples:
# Input: root = [1,2,3,null,null,4,5]
#        1
#       / \
#      2   3
#         / \
#        4   5
# Serialized: "1,2,null,null,3,4,null,null,5,null,null"
# Deserialized: Same tree structure
```

---

### Problem 29: Number of Islands
**Difficulty:** Medium | **LC #200**

Count the number of islands in a 2D grid.

```python
def numIslands(grid: list[list[str]]) -> int:
    if not grid:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    count = 0
    
    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] != '1':
            return
        
        grid[r][c] = '0'  # Mark as visited
        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                count += 1
                dfs(r, c)
    
    return count

# Time: O(m * n), Space: O(m * n) worst case for recursion

# Examples:
# Input: grid = [
#   ["1","1","1","1","0"],
#   ["1","1","0","1","0"],
#   ["1","1","0","0","0"],
#   ["0","0","0","0","0"]
# ]
# Output: 1

# Input: grid = [
#   ["1","1","0","0","0"],
#   ["1","1","0","0","0"],
#   ["0","0","1","0","0"],
#   ["0","0","0","1","1"]
# ]
# Output: 3
```

---

### Problem 30: Clone Graph
**Difficulty:** Medium | **LC #133**

Deep clone a graph.

```python
class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors else []

def cloneGraph(node: Node) -> Node:
    if not node:
        return None
    
    cloned = {}  # original -> clone
    
    def dfs(node):
        if node in cloned:
            return cloned[node]
        
        clone = Node(node.val)
        cloned[node] = clone
        
        for neighbor in node.neighbors:
            clone.neighbors.append(dfs(neighbor))
        
        return clone
    
    return dfs(node)

# Time: O(V + E), Space: O(V)

# Examples:
# Input: adjList = [[2,4],[1,3],[2,4],[1,3]]
#        1 -- 2
#        |    |
#        4 -- 3
# Output: Deep copy of the graph with same structure

# Input: adjList = [[]]
# Output: [[]] (single node with no neighbors)
```

---

### Problem 31: Course Schedule (Cycle Detection)
**Difficulty:** Medium | **LC #207**

Determine if you can finish all courses given prerequisites.

```python
def canFinish(numCourses: int, prerequisites: list[list[int]]) -> bool:
    # Build adjacency list
    graph = [[] for _ in range(numCourses)]
    for course, prereq in prerequisites:
        graph[prereq].append(course)
    
    # 0: unvisited, 1: visiting, 2: visited
    state = [0] * numCourses
    
    def has_cycle(course):
        if state[course] == 1:  # Currently visiting -> cycle
            return True
        if state[course] == 2:  # Already processed
            return False
        
        state[course] = 1
        for next_course in graph[course]:
            if has_cycle(next_course):
                return True
        state[course] = 2
        return False
    
    for course in range(numCourses):
        if has_cycle(course):
            return False
    
    return True

# Time: O(V + E), Space: O(V + E)

# Examples:
# Input: numCourses = 2, prerequisites = [[1,0]]
# Output: True (take course 0, then course 1)

# Input: numCourses = 2, prerequisites = [[1,0],[0,1]]
# Output: False (cycle: 0→1→0)

# Input: numCourses = 4, prerequisites = [[1,0],[2,1],[3,2]]
# Output: True (take 0→1→2→3)
```

---

### Problem 32: Word Ladder
**Difficulty:** Hard | **LC #127**

Find the shortest transformation sequence from beginWord to endWord.

```python
from collections import deque

def ladderLength(beginWord: str, endWord: str, wordList: list[str]) -> int:
    word_set = set(wordList)
    if endWord not in word_set:
        return 0
    
    queue = deque([(beginWord, 1)])
    visited = {beginWord}
    
    while queue:
        word, length = queue.popleft()
        
        if word == endWord:
            return length
        
        # Try changing each character
        for i in range(len(word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                new_word = word[:i] + c + word[i+1:]
                if new_word in word_set and new_word not in visited:
                    visited.add(new_word)
                    queue.append((new_word, length + 1))
    
    return 0

# Time: O(M² * N) where M = word length, N = number of words
# Space: O(M * N)

# Examples:
# Input: beginWord = "hit", endWord = "cog", 
#        wordList = ["hot","dot","dog","lot","log","cog"]
# Output: 5 (hit → hot → dot → dog → cog)

# Input: beginWord = "hit", endWord = "cog",
#        wordList = ["hot","dot","dog","lot","log"]
# Output: 0 (endWord not in wordList)
```

---

## Dynamic Programming

### Problem 33: Climbing Stairs
**Difficulty:** Easy | **LC #70**

Count ways to climb n stairs taking 1 or 2 steps at a time.

```python
def climbStairs(n: int) -> int:
    if n <= 2:
        return n
    
    prev2, prev1 = 1, 2
    for _ in range(3, n + 1):
        current = prev1 + prev2
        prev2 = prev1
        prev1 = current
    
    return prev1

# Time: O(n), Space: O(1)
# This is essentially Fibonacci!

# Examples:
# Input: n = 2
# Output: 2 (1+1 or 2)

# Input: n = 3
# Output: 3 (1+1+1, 1+2, 2+1)

# Input: n = 5
# Output: 8
```

---

### Problem 34: House Robber
**Difficulty:** Medium | **LC #198**

Maximum money you can rob without robbing adjacent houses.

```python
def rob(nums: list[int]) -> int:
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    
    prev2, prev1 = 0, 0
    
    for num in nums:
        current = max(prev1, prev2 + num)
        prev2 = prev1
        prev1 = current
    
    return prev1

# Time: O(n), Space: O(1)

# Examples:
# Input: nums = [2,7,9,3,1]
# Output: 12  (rob houses at index 0, 2, 4: 2+9+1=12)

# Input: nums = [1,2,3,1]
# Output: 4  (rob houses at index 0, 2: 1+3=4)

# Input: nums = [2,1,1,2]
# Output: 4  (rob houses at index 0, 3: 2+2=4)
```

---

### Problem 35: Coin Change
**Difficulty:** Medium | **LC #322**

Find minimum number of coins to make up the amount.

```python
def coinChange(coins: list[int], amount: int) -> int:
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1

# Time: O(amount * len(coins)), Space: O(amount)

# Examples:
# Input: coins = [1,2,5], amount = 11
# Output: 3  (5+5+1 = 11)

# Input: coins = [2], amount = 3
# Output: -1  (impossible)

# Input: coins = [1], amount = 0
# Output: 0
```

---

### Problem 36: Longest Increasing Subsequence
**Difficulty:** Medium | **LC #300**

Find the length of the longest strictly increasing subsequence.

```python
def lengthOfLIS(nums: list[int]) -> int:
    # O(n²) DP solution
    n = len(nums)
    dp = [1] * n  # dp[i] = LIS ending at i
    
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    return max(dp)

# O(n log n) solution using binary search
import bisect

def lengthOfLIS_optimal(nums: list[int]) -> int:
    tails = []  # tails[i] = smallest tail of LIS of length i+1
    
    for num in nums:
        pos = bisect.bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num
    
    return len(tails)

# Time: O(n log n), Space: O(n)

# Examples:
# Input: nums = [10,9,2,5,3,7,101,18]
# Output: 4  (LIS: [2,3,7,101] or [2,5,7,101])

# Input: nums = [0,1,0,3,2,3]
# Output: 4  (LIS: [0,1,2,3])

# Input: nums = [7,7,7,7,7,7,7]
# Output: 1  (all same, strictly increasing)
```

---

### Problem 37: Word Break
**Difficulty:** Medium | **LC #139**

Determine if string can be segmented into dictionary words.

```python
def wordBreak(s: str, wordDict: list[str]) -> bool:
    word_set = set(wordDict)
    n = len(s)
    dp = [False] * (n + 1)
    dp[0] = True  # Empty string
    
    for i in range(1, n + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break
    
    return dp[n]

# Time: O(n² * m) where m = avg word length, Space: O(n)

# Examples:
# Input: s = "leetcode", wordDict = ["leet","code"]
# Output: True  ("leet" + "code")

# Input: s = "applepenapple", wordDict = ["apple","pen"]
# Output: True  ("apple" + "pen" + "apple")

# Input: s = "catsandog", wordDict = ["cats","dog","sand","and","cat"]
# Output: False
```

---

### Problem 38: Unique Paths
**Difficulty:** Medium | **LC #62**

Count unique paths from top-left to bottom-right in a grid.

```python
def uniquePaths(m: int, n: int) -> int:
    # Space-optimized DP
    dp = [1] * n
    
    for _ in range(1, m):
        for j in range(1, n):
            dp[j] += dp[j - 1]
    
    return dp[n - 1]

# Time: O(m * n), Space: O(n)
# Math solution: C(m+n-2, m-1) = (m+n-2)! / ((m-1)! * (n-1)!)

# Examples:
# Input: m = 3, n = 7
# Output: 28

# Input: m = 3, n = 2
# Output: 3  (Right→Right→Down, Right→Down→Right, Down→Right→Right)
```

---

### Problem 39: Edit Distance
**Difficulty:** Medium | **LC #72**

Find minimum operations to convert word1 to word2.

```python
def minDistance(word1: str, word2: str) -> int:
    m, n = len(word1), len(word2)
    
    # dp[i][j] = min ops to convert word1[:i] to word2[:j]
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Base cases
    for i in range(m + 1):
        dp[i][0] = i  # Delete all
    for j in range(n + 1):
        dp[0][j] = j  # Insert all
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # Delete
                    dp[i][j-1],    # Insert
                    dp[i-1][j-1]   # Replace
                )
    
    return dp[m][n]

# Time: O(m * n), Space: O(m * n), can optimize to O(n)

# Examples:
# Input: word1 = "horse", word2 = "ros"
# Output: 3  (horse → rorse → rose → ros)

# Input: word1 = "intention", word2 = "execution"
# Output: 5
```

---

### Problem 40: Maximum Product Subarray
**Difficulty:** Medium | **LC #152**

Find the contiguous subarray with the largest product.

```python
def maxProduct(nums: list[int]) -> int:
    result = nums[0]
    max_prod = min_prod = nums[0]
    
    for i in range(1, len(nums)):
        num = nums[i]
        # Swap if negative (negative * min could become max)
        if num < 0:
            max_prod, min_prod = min_prod, max_prod
        
        max_prod = max(num, max_prod * num)
        min_prod = min(num, min_prod * num)
        result = max(result, max_prod)
    
    return result

# Time: O(n), Space: O(1)

# Examples:
# Input: nums = [2,3,-2,4]
# Output: 6  (subarray [2,3])

# Input: nums = [-2,0,-1]
# Output: 0  (subarray [0])

# Input: nums = [-2,3,-4]
# Output: 24  (entire array: -2 * 3 * -4)
```

---

## Binary Search

### Problem 41: Search in Rotated Sorted Array
**Difficulty:** Medium | **LC #33**

Search for a target in a rotated sorted array.

```python
def search(nums: list[int], target: int) -> int:
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if nums[mid] == target:
            return mid
        
        # Left half is sorted
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        # Right half is sorted
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    
    return -1

# Time: O(log n), Space: O(1)

# Examples:
# Input: nums = [4,5,6,7,0,1,2], target = 0
# Output: 4

# Input: nums = [4,5,6,7,0,1,2], target = 3
# Output: -1

# Input: nums = [1], target = 0
# Output: -1
```

---

### Problem 42: Find Minimum in Rotated Sorted Array
**Difficulty:** Medium | **LC #153**

Find the minimum element in a rotated sorted array.

```python
def findMin(nums: list[int]) -> int:
    left, right = 0, len(nums) - 1
    
    while left < right:
        mid = (left + right) // 2
        
        if nums[mid] > nums[right]:
            # Minimum is in right half
            left = mid + 1
        else:
            # Minimum is in left half (including mid)
            right = mid
    
    return nums[left]

# Time: O(log n), Space: O(1)

# Examples:
# Input: nums = [3,4,5,1,2]
# Output: 1

# Input: nums = [4,5,6,7,0,1,2]
# Output: 0

# Input: nums = [11,13,15,17]
# Output: 11  (not rotated)
```

---

### Problem 43: Search a 2D Matrix
**Difficulty:** Medium | **LC #74**

Search for a value in a row-wise and column-wise sorted matrix.

```python
def searchMatrix(matrix: list[list[int]], target: int) -> bool:
    if not matrix:
        return False
    
    m, n = len(matrix), len(matrix[0])
    left, right = 0, m * n - 1
    
    while left <= right:
        mid = (left + right) // 2
        row, col = mid // n, mid % n
        val = matrix[row][col]
        
        if val == target:
            return True
        elif val < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return False

# Time: O(log(m*n)), Space: O(1)

# Examples:
# Input: matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 3
# Output: True

# Input: matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 13
# Output: False
```

---

### Problem 44: Median of Two Sorted Arrays
**Difficulty:** Hard | **LC #4**

Find the median of two sorted arrays in O(log(m+n)) time.

```python
def findMedianSortedArrays(nums1: list[int], nums2: list[int]) -> float:
    # Ensure nums1 is smaller
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1
    
    m, n = len(nums1), len(nums2)
    left, right = 0, m
    
    while left <= right:
        partition1 = (left + right) // 2
        partition2 = (m + n + 1) // 2 - partition1
        
        maxLeft1 = float('-inf') if partition1 == 0 else nums1[partition1 - 1]
        minRight1 = float('inf') if partition1 == m else nums1[partition1]
        
        maxLeft2 = float('-inf') if partition2 == 0 else nums2[partition2 - 1]
        minRight2 = float('inf') if partition2 == n else nums2[partition2]
        
        if maxLeft1 <= minRight2 and maxLeft2 <= minRight1:
            if (m + n) % 2 == 0:
                return (max(maxLeft1, maxLeft2) + min(minRight1, minRight2)) / 2
            else:
                return max(maxLeft1, maxLeft2)
        elif maxLeft1 > minRight2:
            right = partition1 - 1
        else:
            left = partition1 + 1
    
    return 0.0

# Time: O(log(min(m,n))), Space: O(1)

# Examples:
# Input: nums1 = [1,3], nums2 = [2]
# Output: 2.0  (merged: [1,2,3], median = 2)

# Input: nums1 = [1,2], nums2 = [3,4]
# Output: 2.5  (merged: [1,2,3,4], median = (2+3)/2)
```

---

## Stack & Queue

### Problem 45: Daily Temperatures
**Difficulty:** Medium | **LC #739**

Find days until a warmer temperature for each day.

```python
def dailyTemperatures(temperatures: list[int]) -> list[int]:
    n = len(temperatures)
    result = [0] * n
    stack = []  # Stack of indices
    
    for i in range(n):
        while stack and temperatures[i] > temperatures[stack[-1]]:
            prev_idx = stack.pop()
            result[prev_idx] = i - prev_idx
        stack.append(i)
    
    return result

# Time: O(n), Space: O(n)

# Examples:
# Input: temperatures = [73,74,75,71,69,72,76,73]
# Output: [1,1,4,2,1,1,0,0]
# Explanation: For 73°, next warmer is 74° (1 day)
#              For 75°, next warmer is 76° (4 days)

# Input: temperatures = [30,40,50,60]
# Output: [1,1,1,0]
```

---

### Problem 46: Largest Rectangle in Histogram
**Difficulty:** Hard | **LC #84**

Find the largest rectangle area in a histogram.

```python
def largestRectangleArea(heights: list[int]) -> int:
    stack = []  # Stack of indices
    max_area = 0
    
    for i, h in enumerate(heights + [0]):  # Add 0 to flush stack
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, height * width)
        stack.append(i)
    
    return max_area

# Time: O(n), Space: O(n)

# Examples:
# Input: heights = [2,1,5,6,2,3]
# Output: 10  (rectangle of height 5, width 2 at indices 2-3)

# Input: heights = [2,4]
# Output: 4  (rectangle of height 2, width 2)
```

---

### Problem 47: Implement Queue using Stacks
**Difficulty:** Easy | **LC #232**

Implement a FIFO queue using two stacks.

```python
class MyQueue:
    def __init__(self):
        self.stack_in = []
        self.stack_out = []
    
    def push(self, x: int) -> None:
        self.stack_in.append(x)
    
    def pop(self) -> int:
        self._transfer()
        return self.stack_out.pop()
    
    def peek(self) -> int:
        self._transfer()
        return self.stack_out[-1]
    
    def empty(self) -> bool:
        return not self.stack_in and not self.stack_out
    
    def _transfer(self):
        if not self.stack_out:
            while self.stack_in:
                self.stack_out.append(self.stack_in.pop())

# Amortized O(1) for all operations

# Example:
# queue = MyQueue()
# queue.push(1)      # stack_in: [1]
# queue.push(2)      # stack_in: [1,2]
# queue.peek()       # returns 1, stack_out: [2,1]
# queue.pop()        # returns 1, stack_out: [2]
# queue.empty()      # returns False
```

---

## Math & Bit Manipulation

### Problem 48: Reverse Integer
**Difficulty:** Medium | **LC #7**

Reverse digits of a 32-bit signed integer.

```python
def reverse(x: int) -> int:
    INT_MAX, INT_MIN = 2**31 - 1, -2**31
    
    sign = 1 if x >= 0 else -1
    x = abs(x)
    
    result = 0
    while x:
        digit = x % 10
        x //= 10
        
        # Check overflow before adding
        if result > (INT_MAX - digit) // 10:
            return 0
        
        result = result * 10 + digit
    
    return sign * result

# Time: O(log x), Space: O(1)

# Examples:
# Input: x = 123
# Output: 321

# Input: x = -123
# Output: -321

# Input: x = 120
# Output: 21

# Input: x = 1534236469
# Output: 0  (overflow)
```

---

### Problem 49: Number of 1 Bits
**Difficulty:** Easy | **LC #191**

Count the number of 1 bits in an integer.

```python
def hammingWeight(n: int) -> int:
    count = 0
    while n:
        count += n & 1
        n >>= 1
    return count

# Brian Kernighan's algorithm (faster)
def hammingWeight_v2(n: int) -> int:
    count = 0
    while n:
        n &= (n - 1)  # Removes rightmost 1 bit
        count += 1
    return count

# Time: O(log n) or O(number of 1 bits), Space: O(1)

# Examples:
# Input: n = 11 (binary: 1011)
# Output: 3

# Input: n = 128 (binary: 10000000)
# Output: 1

# Input: n = 2147483645 (binary: 1111111111111111111111111111101)
# Output: 30
```

---

### Problem 50: Single Number
**Difficulty:** Easy | **LC #136**

Find the element that appears only once (all others appear twice).

```python
def singleNumber(nums: list[int]) -> int:
    result = 0
    for num in nums:
        result ^= num  # XOR: a ^ a = 0, a ^ 0 = a
    return result

# Time: O(n), Space: O(1)

# Examples:
# Input: nums = [4,1,2,1,2]
# Output: 4  (1 and 2 appear twice, 4 appears once)

# Input: nums = [2,2,1]
# Output: 1

# Input: nums = [1]
# Output: 1
```

---

## Bonus: Common Patterns Cheat Sheet

### Two Pointers
```python
# Opposite ends
left, right = 0, len(arr) - 1
while left < right:
    # process
    left += 1  # or right -= 1

# Same direction (fast/slow)
slow = fast = 0
while fast < len(arr):
    # process
    fast += 1
```

### Sliding Window
```python
left = 0
for right in range(len(arr)):
    # expand window
    while window_invalid:
        # shrink from left
        left += 1
    # update result
```

### Binary Search
```python
left, right = 0, len(arr) - 1
while left <= right:  # or left < right
    mid = (left + right) // 2
    if condition:
        right = mid - 1  # or right = mid
    else:
        left = mid + 1
```

### BFS Template
```python
from collections import deque
queue = deque([start])
visited = {start}
while queue:
    node = queue.popleft()
    for neighbor in get_neighbors(node):
        if neighbor not in visited:
            visited.add(neighbor)
            queue.append(neighbor)
```

### DFS Template
```python
def dfs(node, visited):
    if node in visited:
        return
    visited.add(node)
    for neighbor in get_neighbors(node):
        dfs(neighbor, visited)
```

### DP Template
```python
# Bottom-up
dp = [base_case] * (n + 1)
for i in range(1, n + 1):
    dp[i] = recurrence(dp[i-1], ...)

# Top-down with memoization
@lru_cache(maxsize=None)
def solve(state):
    if base_case:
        return base_value
    return recurrence(solve(subproblem))
```

### Monotonic Stack
```python
stack = []
for i, val in enumerate(arr):
    while stack and arr[stack[-1]] < val:  # or >
        idx = stack.pop()
        # process idx
    stack.append(i)
```

---

## Time Complexity Quick Reference

| Algorithm | Time | Space |
|-----------|------|-------|
| Binary Search | O(log n) | O(1) |
| Two Pointers | O(n) | O(1) |
| Sliding Window | O(n) | O(k) |
| BFS/DFS | O(V + E) | O(V) |
| Sorting | O(n log n) | O(n) |
| Heap Operations | O(log n) | O(n) |
| Hash Table | O(1) avg | O(n) |
| DP (1D) | O(n) | O(n) |
| DP (2D) | O(n²) | O(n²) |

---

