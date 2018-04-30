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

```
