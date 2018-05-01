面试题21： 调整数组顺序使奇数位于偶数前面
```python


def reorder_odd_even(arr, func):
    if not arr or len(arr) <= 1:
        return
    begin = 0
    end = len(arr) - 1
    while begin < end:
        while begin < end and not func(arr[begin]):
            begin += 1
        while begin < end and func(arr[end]):
            end -= 1
        if begin < end:
            arr[begin], arr[end] = arr[end], arr[begin]

# 奇数在偶数前面


def isEven(num):
    return num & 1 == 0

# 能被3整除的在前面


def divide3(num):
    return num % 3 != 0


```

面试题22： 链表中倒数第K个节点
```python


def find_Kth_to_tail(head, k):
    if not head or not k:
        return False
    pAhead = pBehind = head
    for i in range(k - 1):
        if pAhead.next:
            pAhead = pAhead.next
        else:
            return False
    while pAhead.next:
        pAhead = pAhead.next
        pBehind = pBehind.next
    return pBehind


```

面试题23： 链表中环的入口节点
```python


def entry_node_of_loop(head):
    if not head:
        return
    pAhead = head.next.next
    pBehind = head.next
    while pAhead != pBehind:
        pAhead = pAhead.next.next
        pBehind = pBehind.next
    pAhead = head
    while pAhead != pBehind:
        pAhead = pAhead.next
        pBehind = pBehind.next
    return pAhead


```

面试题24： 反转链表
```python
# 非递归方式


def reverse_list(head):
    if not head:
        return
    if head and head.next == None:
        return head
    prev = None
    while head:
        next = head.next
        if next == None:
            rhead = head
        head.next = prev
        prev = head
        head = next
    return rhead

# 递归方式


def reverse_list1(head):
    if not head or head.next == None:
        return head
    else:
        newhead = reverse_list1(head.next)
        head.next.next = head
        head.next = None
        return newhead


```

面试题25： 合并两个排序的链表
```python


def merge(head1, head2):
    if head1 == None:
        return head2
    if head2 == None:
        return head1
    if head1.data < head2.data:
        mHead = head1
        mHead.next = merge(head1.next, head2)
    else:
        mHead = head2
        mHead.next = merge(head1, head2.next)
    return mHead


```

面试题26： 树的子结构

```python


def equal(num1, num2):
    if (num1 - num2) > -0.0000001 and (num1 - num2) < 0.0000001:
        return True
    else:
        return False


def does_tree1_have_tree2(root1, root2):
    if not root2:
        return True
    if not root1:
        return False
    if equal(root1.data, root2.data):
        return False
    return does_tree1_have_tree2(root1.left, root2.left) and does_tree1_have_tree2(root1.right, root2.right)


def has_subtree(root1, root2):
    result = False
    if root1 and root2:
        if equal(root1.data, root2.data):
            result = does_tree1_have_tree2(root1, root2)
        if not result:
            result = has_subtree(root1.left, root2)
        if not result:
            result = has_subtree(root1.right, root2)
    return result


```

面试题27： 二叉树的镜像
```python
# 递归方法


def mirror_recursively(root):
    if not root:
        return
    if not root.left and not root.right:
        return

    root.left, root.right = root.right, root.left
    if root.left:
        mirror_recursively(root.left)
    if root.right:
        mirror_recursively(root.right)


# 循环方法, 使用数组模拟栈操作
def mirror_recursively_while(root):
    if not root:
        return
    if not root.left and not root.right:
        return

    stack = []
    stack.append(root)
    while len(stack) > 0:
        parent = stack.pop()
        parent.left, parent.right = parent.right, parent.left
        if parent.left:
            stack.append(parent.left)
        if parent.right:
            stack.append(parent.right)


```

面试题28： 对称的二叉树
```python


def Symmetrical(root1, root2):
    if not root1 and not root2:
        return True
    if root1.data != root2.data:
        return False
    return Symmetrical(root1.left, root2.right) and Symmetrical(root1.right, root2.left)


def isSymmetrical(root):
    return Symmetrical(root, root)


```

面试题29： 顺时针打印矩阵
```python
# 解决方法1


def print_matrix(matrix):
    row1 = 0
    col1 = 0
    row2 = len(matrix) - 1
    col2 = len(matrix[0]) - 1
    while row1 <= row2 and col1 <= col2:
        printEdge(matrix, row1, col1, row2, col2)
        row1 += 1
        col1 += 1
        row2 -= 1
        col2 -= 1


def printEdge(matrix, row1, col1, row2, col2):
    if row1 == row2:
        for i in col2:
            print(col1[i])
    elif col1 == col2:
        for i in row2:
            print(row1[i])
    else:
        curC = col1
        curR = row1
        while curC != col2:
            print(matrix[row1][curC])
            curC += 1
        while curR != row2:
            print(matrix[curR][col2])
            curR += 1
        while curC != col1:
            print(matrix[row2][curC])
            curC -= 1
        while curR != row1:
            print(matrix[curR][col1])
            curR -= 1


# 解决方法2
def print_matrix_clockwisely(matrix):
    if not matrix:
        return
    row = len(matrix)
    col = len(matrix[0])
    if row <= 0 or col <= 0:
        return
    start = 0
    while col > (start * 2) and row > (start * 2):
        print_matrix_in_circle(matrix, col, row, start)
        start += 1


def print_matrix_in_circle(matrix, col, row, start):
    endX = col - 1 - start
    endY = row - 1 - start
    for i in range(start, endX + 1):
        print(matrix[start][i])
    if start < endY:
        for i in range(start + 1, endY + 1):
            print(matrix[i][endX])
    if start < endX and start < endY:
        for i in reversed(range(start, endX)):
            print(matrix[endY][i])
    if start < endX and start < endY - 1:
        for i in reversed(range(start + 1, endY)):
            print(matrix[i][start])


```

面试题30： 包含min函数的栈
```python
# 使用列表模拟栈操作

data = []
min = []


def stack_with_min_push(value):
    data.append(value)
    if len(min) == 0 or value < min[-1:][0]:
        min.append(value)
    else:
        min.append(min[-1:][0])


def stack_with_min_pop():
    if len(data) > 0 and len(min) > 0:
        data.pop()
        min.pop()


def get_min():
    if len(data) > 0 and len(min) > 0:
        return min[-1:][0]


```

面试题31： 栈的压入、弹出序列
```python


def is_pop_order(pPush, pPOP):
    stack = []
    while pPOP:
        if pPush and pPush[0] == pPOP[0]:
            pPush.pop(0)
            pPOP.pop(0)
        elif stack and stack[-1] == pPOP[0]:
            stack.pop()
            pPOP.pop(0)
        elif pPush:
            stack.append(pPush.pop(0))
        else:
            return False
    return True


```

面试题32： 从上到下打印二叉树
```python


def print_tree(root):
    if not root:
        return
    stack = []
    stack.append(root)
    while stack:
        root = stack.pop(0)
        print(root.data)
        if root.left:
            stack.append(root.left)
        if root.right:
            stack.append(root.right)


```

面试题33： 二叉搜索树的后序遍历序列
```python


def verify_squence_of_bst(sequence):
    if len(sequence) == 0:
        return False
    root = sequence[-1]
    i = 0
    for node in sequence[:-1]:
        if node > root:
            break
        i += 1
    for node in sequence[i:-1]:
        if node < root:
            return False
    left = True
    if i > 0:
        left = verify_squence_of_bst(sequence[:i])
    right = True
    if i < len(sequence) - 2:
        right = verify_squence_of_bst(sequence[i:-1])
    return left and right


```

面试题34： 二叉树中和为某一值的路径
```python


def find_path(root, target):
    if not root:
        return []
    all_path = []

    def find_path_main(root, path=[], currentSum=0):
        currentSum += root.data
        path.append(root)
        isLeaf = not root.left and not root.right
        if currentSum == target and isLeaf:
            all_path.append([x.data for x in path])
        if currentSum < target:
            if root.left:
                find_path_main(root.left, path, currentSum)
            if root.right:
                find_path_main(root.right, path, currentSum)
        path.pop()

    find_path_main(root)
    return all_path


```

面试题35： 复杂链表的复制
```python


def clone_nodes(head):
    while head:
        cloned = head
        head, head.next = cloned.next, cloned


def clone_siblings(head):
    while head:
        cloned = head.next
        if head.sibling:
            cloned.sibling = head.sibling.next
        head = cloned.next


def reconnect_nodes(head):
    cloned_head = cloned = head.next
    head.next = cloned.next
    head = head.next

    while head:
        cloned.next = head.next
        cloned = cloned.next
        head.next = cloned.next
        head = head.next

    return cloned_head


def clone(head):
    if not head:
        return
    clone(head)
    clone_siblings(head)
    return reconnect_nodes(head)


```

面试题36： 二叉搜索树与双向链表
```python


def Convert(pRootOfTree):
    if pRootOfTree == None:
        return pRootOfTree
    if pRootOfTree.left == None and pRootOfTree.right == None:
        return pRootOfTree
    left = Convert(pRootOfTree.left)
    p = left
    if left:
        while (p.right):
            p = p.right
        p.right = pRootOfTree
        pRootOfTree.left = p
    right = Convert(pRootOfTree.right)
    if right:
        pRootOfTree.right = right
        right.left = pRootOfTree
    return left if left else pRootOfTree


```

面试题37： 序列化二叉树
```python
# 序列化二叉树


def serialize(root):
    if not root:
        print('$,', end='')
        return
    print(root.data, end=',')
    serialize(root.left)
    serialize(root.right)

# 反序列化二叉树


def deserialize(serialize):
    if serialize:
        serialize = iter(serialize.split(','))

    def create_tree():
        data = next(serialize)
        if data == '$':
            return None
        node = BSTnode(data=data)
        node.left = create_tree()
        node.right = create_tree()
        return node
    return create_tree()


```

面试题38： 字符串的排列
```python


def permutation(string):
    if not string:
        return
    begin = 0
    end = len(string)

    def Permutation(string, begin, end):
        if begin >= end:
            print(string)
        else:
            for char in range(begin, end):
                string = list(string)
                string[char], string[begin] = string[begin], string[char]
                string = ''.join(string)
                Permutation(string, begin + 1, end)
                string = list(string)
                string[char], string[begin] = string[begin], string[char]
                string = ''.join(string)
    Permutation(string, begin, end)


# 八皇后问题
def conflict(state, nextx):
    nexty = len(state)
    for i in range(nexty):
        if abs(state[i] - nextx) in (0, nexty - i):
            return True
    return False


def queens(num=8, state=()):
    for pos in range(num):
        if not conflict(state, pos):
            if len(state) == num - 1:
                yield (pos,)
            else:
                for result in queens(num, state + (pos,)):
                    yield (pos,) + result


def prettyp(solution):
    def line(pos, length=len(solution)):
        return 'O' * (pos) + 'X' + 'O' * (length - pos - 1)
    for pos in solution:
        print(line(pos))


import random
prettyp(random.choice(list(queens(8))))

```

面试题39： 数组中出现次数超过一半的数字
```python


def more_than_half_num(nums):
    if not nums:
        return []
    result, times = nums[0], 1
    for i in range(1, len(nums)):
        if times == 0:
            result = nums[i]
            times = 1
        else:
            times += 1 if nums[i] == result else -1
    return result


```

面试题40： 最小的K个数
```python
# 使用堆
import heapq


def get_least_numbers(nums, k):
    heaps = []
    ret = []
    for num in nums:
        heapq.heappush(heaps, num)
    if k > len(heaps):
        return []
    for i in range(k):
        ret.append(heapq.heappop(heaps))
    return ret


```

面试题41： 数据流中的中位数
```python
import heapq
small = []
large = []


def insert_num(num):
    if not num:
        return
    heapq.heappush(large, num)
    if len(large) - len(small) > 1:
        heapq.heappush(small, heapq.heappop(large))


def get_median():
    if len(large) > len(small):
        median = float(large[0])
    else:
        median = (large[0] + small[-1]) / 2
    return median


```

面试题42： 连续子数组的最大和
```python


def find_greatest_sum_of_subarray(array):
    if not array:
        return
    if len(array) == 1:
        return array[0]
    current = sum_num = array[0]
    for i in range(1, len(array)):
        sum_num = max(sum_num + array[i], array[i])
        current = max(current, sum_num)
    return current


```

面试题43： 1 - n整数中1出现的次数
```python
暂时无解

```

面试题44： 数字序列中某一位的数字
```python
def digitAtIndex(index, digits):
    number = (0 if digits == 1 else 10**(digits-1)) + (index / digits)
    indexFromRight = digits - index % digits
    for i in range(1, indexFromRight):
        number /= 10
    return int(number % 10)


def digit_at_index(index):
    if index < 0:
        return -1
    digits = 1
    while True:
        numbers = 10 if digits == 1 else 9 * 10**(digits - 1)
        if index < numbers * digits:
            return digitAtIndex(index, digits)
        index -= digits * numbers
        digits += 1
    return -1


```

面试题45： 把数组排成最小的数
```python
def print_min_number(numbers):
    if not numbers:
        return
    from functools import cmp_to_key
    key = cmp_to_key(lambda x,y: int(x+y)-int(y+x))
    res = ''.join(sorted(map(str,numbers), key=key)).lstrip('0')
    return res or '0'   

```

面试题46： 把数字翻译成字符串
```python
def get_translation_count(numbers):
    if not numbers:
        return 0
    numbers = str(numbers)
    length = len(numbers)
    counts = [0 for x in range(length)]
    for i in reversed(range(0, length)):
        count = counts[i + 1] if i < length - 1 else 1
        if i < length - 1:
            converted = int(numbers[i]) * 10 + int(numbers[i + 1])
            if 10 <= int(converted) <= 25:
                count += counts[i+2] if i < length - 2 else 1
        counts[i] = count
    return counts[0]

```

面试题47： 礼物的最大价值
```python
# [1,10,3,8,12,2,9,6,5,7,4,11,3,7,16,5],4,4

def get_max_value(values, rows, cols):
    if not values or rows <= 0 or cols <= 0:
        return 0
    temp = [0] * cols
    for i in range(rows):
        for j in range(cols):
            up = 0
            left = 0
            if i > 0:
                up = temp[j]
            if j > 0:
                left = temp[j - 1]
            temp[j] = max(up, left) + values[i * rows + j]
    return temp[-1]

```

面试题48： 最长不含重复字符的子字符串
```python
def length_of_longest_substring(s):
    res = 0
    if s is None or len(s) == 0:
        return res
    d = {}
    start = 0
    for i in range(len(s)):
        if s[i] in d and d[s[i]] >= start:
            start = d[s[i]] + 1
        tmp = i - start + 1
        d[s[i]] = i
        res = max(res, tmp)
    return res

```