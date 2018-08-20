面试题1 - - 面试题20：
```python
等待补充
```

#### 面试题2： 实现Singleton模式

> 设计一个类，我们只能生成该类的一个实例

```python
import threading

def synchronize(func):
    func.__lock__ = threading.Lock()

    def synced_func(*args, **kwargs):
        with func.__lock__:
            return func(*args, **kwargs)

    return synced_func


# 类方法
class SinletonClass(object):

    @synchronize
    def __new__(cls, *args, **kwargs):
        if not getattr(cls, '_instance'):
            cls._instance = super(SinletonClass, cls).__new__(cls, *args, **kwargs)
        return cls._instance


# 装饰器方法
def SinletonDecorator(cls):
    _instance = {}

    @synchronize
    def get_instance(*args, **kwargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kwargs)
        return _instance
    
    return get_instance
```

#### 面试题3：数组中重复的数字

> 题目一：
>
> 在一个长度为n的数组里有所有数字都在0~n-1的范围内，数组中某些数字是重复的，但不知道有几个数字重复了，也不知道每个数字重复了几次，请找出数组中任意一个重复的数字，例如，如果输入长度为7的数组 [ 2, 3, 1, 0, 2, 5, 3 ] ，那么对应的输出是重复的数字2或者3

```python
def duplicate(arr):
    if not arr:
        return False

    length = len(arr)

    for i in range(length):
        if arr[i] < 0 or arr[i] > length - 1:
            return False

    for i in range(length):
        while arr[i] != i:
            if arr[i] == arr[arr[i]]:
                return arr[i]
            temp = arr[i]
            arr[i], arr[temp] = arr[temp], arr[i]
    return False
```

> 题目二：
>
> 在一个长度为n+1的数组里的所有数字都在1-n的范围内，所以数组中至少有一个数字是重复的，请找出数组中任意一个重复的数字，但不能修改输入的数组，例如，如果输入长度为8 的数组 [ 2, 3, 5, 4, 3, 2, 6, 7 ] 那么对应的输出是重复的数字2或者

```PYTHON
def duplicate2(arr):
    if not arr:
        return False

    length = len(arr)

    for i in range(length):
        if arr[i] < 0 or arr[i] >= length:
            return False

    new_arr = [None] * length

    for i in range(length):
        if new_arr[arr[i]] is None:
            new_arr[arr[i]] = arr[i]
        else:
            return arr[i]

    return False
```

#### 面试题4：二维数组中的查找

> 在一个二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序，请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。

```python
def Find(target, array):
    if not array: 
        return False
    col = len(array[0]) - 1
    cur_col = 0
    cur_row = len(array) - 1
    while cur_col <= col and cur_row >= 0:
        if array[cur_row][cur_col] == target:
            return True
        if array[cur_row][cur_col] > target:
            cur_row -= 1
        if array[cur_row][cur_col] < target:
            cur_col += 1
    return False
```

#### 面试题5：替换空格

> 请实现一个函数，把字符串中的每个空格替换成“%20”，例如，输入“We are happy”，则输出“We%20are%20happy”

```python
def replace_blank(string):
    if not string:
        return ""
    new_string = ''
    for i in string:
        if i == ' ':
            new_string += '%20'
        else:
            new_string += i
    return new_string
```

#### 面试题6：从尾到头打印链表

> 输入一个链表的头节点，从尾到头反过来打印出每个节点的值。

```python
class LinkNode:
    def __init__(self, val=None, next=None):
        self.val = val
        self.next = next

def PrintListReversingly(head):
    if head:
        if head.next:
            PrintListReversingly(head.next)
        print(head.val)
```

#### 面试题7：重建二叉树

> 输入某二叉树的前序遍历和中序遍历的结果，请重建该二叉树，假设输入的前序遍历和中序遍历的结果中都不含重复的数字，例如，输入前序遍历序列 [ 1, 2, 4, 7, 3, 5, 6, 8 ] 和中序遍历序列 [ 4, 7, 2, 1, 5, 3, 8, 6 ] ,则重建以下二叉树
>
> ​			 ①
>
> ​		      /	       \
>
> ​		②		  ③
>
> ​             /		       /       \
>
> ​	④		  ⑤		  ⑥
>
> ​             \			       /
>
> ​		⑦		  ⑧

```python
def reConstructBinaryTree(pre, tin):
    if not pre or not tin:
        return None
    root = TreeNode(pre.pop(0))
    index = tin.index(root.val)
    root.left = reConstructBinaryTree(pre, tin[:index])
    root.right = reConstructBinaryTree(pre, tin[index+1:])
    return root
```

#### 面试题8：二叉树的下一个节点

> 给定一个二叉树和其中的一个结点，请找出中序遍历顺序的下一个结点并且返回。注意，树中的结点不仅包含左右子结点，同时包含指向父结点的指针。 

```python
class TreeLinkNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
        self.next = None

def GetNext(pNode):
    if not pNode:
        return

    elif pNode.right != None:
        pNode = pNode.right
        while pNode.left != None:
            pNode = pNode.left
        return pNode

    elif pNode.next != None and pNode.next.right == pNode:
        while pNode.next != None and pNode.next.left != pNode:
            pNode = pNode.next
        return pNode.next

    else:
        return pNode.next
```

#### 面试题9：用两个栈实现队列 

> 题目1：用两个栈来实现一个队列，完成队列的Push和Pop操作。 队列中的元素为int类型。 

```python
class StackToQueue:
    def __init__(self):
        self.stackA = []
        self.stackB = []

    def push(self, node):
        self.stackA.append(node)

    def pop(self):
        if self.stackB:
            return self.stackB.pop()
        elif not self.stackA:
            return None
        else:
            while self.stackA:
                self.stackB.append(self.stackA.pop())
            return self.stackB.pop()
```

> 题目2：用两个队列实现一个栈

```python
class QueueToStack:
    def __init__(self):
        self.queueA = []
        self.queueB = []

    def push(self, item):
        if self.queueB:
            self.queueB.append(item)
        else:
            self.queueA.append(item)


    def pop(self):
        if not self.queueA and not self.queueB:
            return None

        if self.queueA:
            while len(self.queueA) > 1:
                self.queueB.append(self.queueA.pop(0))
            return self.queueA.pop()
        else:
            while len(self.queueB) > 1:
                self.queueA.append(self.queueB.pop(0))
            return self.queueB.pop()
```

#### 面试题10：斐波那切数列 

> 大家都知道斐波那契数列，现在要求输入一个整数n，请你输出斐波那契数列的第n项。
> n<=39

```python
 def Fibonacci(n):
    if n > 39:
        return 0
    res = [0,1]
    while len(res) <= n:
        res.append(res[-1] + res[-2])
    return res[n]
```

#### 面试题11：旋转数组的最小数字 

> 把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。 输入一个非减排序的数组的一个旋转，输出旋转数组的最小元素。 例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。 NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。 

```python
def minNumberInRotateArray(rotateArray):
    if not rotateArray:
        return 0
    index = rotateArray[0]
    while index >= rotateArray[-1]:
        rotateArray.insert(0, rotateArray[-1])
        rotateArray.pop()
    return rotateArray[0]
```

#### 面试题12：矩阵中的路径 

#### 面试题13：机器人的运动范围 

#### 面试题14：剪绳子 

#### 面试题15：二进制中1的个数 

> 输入一个整数，输出该数二进制表示中1的个数。其中负数用补码表示。

```python
def NumberOf1(n):
    return bin(n).count('1') if n >= 0 else bin(2**32 + n).count('1')
```

#### 面试题16：数值的整数次方 

#### 面试题17：打印从1到最大的n位数 

#### 面试题18：删除链表的节点 

#### 面试题19：正则表达式匹配 

#### 面试题20：表示数值的字符串 

#### 面试题21： 调整数组顺序使奇数位于偶数前面

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

#### 面试题22： 链表中倒数第K个节点

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

#### 面试题23： 链表中环的入口节点

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

#### 面试题24： 反转链表

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

#### 面试题25： 合并两个排序的链表

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

#### 面试题26： 树的子结构

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

#### 面试题27： 二叉树的镜像

```python
# 递归方法


class Solution:
    # 返回镜像树的根节点
    def Mirror(self, root):
        if not root:
            return
        if not root.left and not root.right:
            return

        root.left, root.right = root.right, root.left
        if root.left:
            self.Mirror(root.left)
        if root.right:
            self.Mirror(root.right)


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

#### 面试题28： 对称的二叉树

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

#### 面试题29： 顺时针打印矩阵

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

#### 面试题30： 包含min函数的栈

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

#### 面试题31： 栈的压入、弹出序列

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

#### 面试题32： 从上到下打印二叉树

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

#### 面试题33： 二叉搜索树的后序遍历序列

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

#### 面试题34： 二叉树中和为某一值的路径

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

#### 面试题35： 复杂链表的复制

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

#### 面试题36： 二叉搜索树与双向链表

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

#### 面试题37： 序列化二叉树

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

#### 面试题38： 字符串的排列

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

#### 面试题39： 数组中出现次数超过一半的数字

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

#### 面试题40： 最小的K个数

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

#### 面试题41： 数据流中的中位数

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

#### 面试题42： 连续子数组的最大和

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

#### 面试题43： 1 - n整数中1出现的次数

```python
def NumberOf1Between1AndN_Solution(n):
    count=0
    for i in range(1,n+1):
        for j in str(i):
            if j=="1":
                count+=1
    return count
```

#### 面试题44： 数字序列中某一位的数字

```python
def digitAtIndex(index, digits):
    number = (0 if digits == 1 else 10**(digits - 1)) + (index / digits)
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

#### 面试题45： 把数组排成最小的数

```python
def print_min_number(numbers):
    if not numbers:
        return
    from functools import cmp_to_key
    key = cmp_to_key(lambda x, y: int(x + y) - int(y + x))
    res = ''.join(sorted(map(str, numbers), key=key)).lstrip('0')
    return res or '0'
```

#### 面试题46： 把数字翻译成字符串

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
                count += counts[i + 2] if i < length - 2 else 1
        counts[i] = count
    return counts[0]
```

#### 面试题47： 礼物的最大价值

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

#### 面试题48： 最长不含重复字符的子字符串

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

#### 面试题49： 丑数

```python
def get_ugly_number(index):
    if not index:
        return 0
    ugly_number = [1] * index
    next_index = 1
    m2 = m3 = m5 = 1
    while next_index < index:
        ugly_number[next_index] = min(m2 * 2, m3 * 3, m5 * 5)
        while m2 * 2 <= ugly_number[next_index]:
            m2 += 1
        while m3 * 3 <= ugly_number[next_index]:
            m3 += 1
        while m5 * 5 <= ugly_number[next_index]:
            m5 += 1
        next_index += 1
    return ugly_number[next_index - 1]
```

#### 面试题50： 第一个只出现一次的字符

```python
def first_not_repeating_char(string):
    if not string:
        return
    temp = dict([(x, 0) for x in range(26)])
    for s in string:
        # ord('a') == 97
        temp[ord(s) - 97] += 1
    for s in string:
        if temp[ord(s) - 97] == 1:
            res = s
            break
    return res
```

#### 面试题51： 数组中的逆序对

```python
class Solution:
    def InversePairs(self, data):
        self.count = 0

        def MergeSort(lists):
            if len(lists) <= 1:
                return lists
            num = len(lists) >> 1
            left = MergeSort(lists[:num])
            right = MergeSort(lists[num:])
            R = L = 0
            LL, LR = len(left), len(right)
            result = []
            while L < LL and R < LR:
                if left[L] < right[R]:
                    result.append(left[L])
                    L += 1
                else:
                    result.append(right[R])
                    R += 1
                    self.count += LL - L
            result += right[R:]
            result += left[L:]
            return result
        MergeSort(data)
        return self.count
```

#### 面试题52： 两个链表中的第一个公共节点

```python
def find_first_common_node(pHead1, pHead2):
    if not pHead1 or not pHead2:
        return None
    pa = pHead1
    pb = pHead2
    while(pa != pb):
        pa = pHead2 if pa is None else pa.next
        pb = pHead1 if pb is None else pb.next
    return pa
```

#### 面试题53： 在排序数组中查找数字

```python
def get_k(data, k, start, end, first):
    if start > end:
        return -1
    mid = (end + start) >> 1
    mid_data = data[mid]
    if mid_data == k:
        if first:
            if mid > 0 and data[mid - 1] != k or mid == 0:
                return mid
            else:
                end = mid - 1
        else:
            if mid < len(data) - 1 and data[mid + 1] != k or mid == len(data) - 1:
                return mid
            else:
                start = mid + 1
    elif mid_data > k:
        end = mid - 1
    else:
        start = mid + 1
    return get_k(data, k, start, end, first)


def get_number_of_k(data, k):
    if not data or not k:
        return
    number = 0
    length = len(data)
    if data and length > 0:
        first = get_k(data, k, 0, length - 1, first=True)
        last = get_k(data, k, 0, length - 1, first=False)
        if first > -1 and last > -1:
            number = last - first + 1
    return number
```

#### 面试题54： 二叉搜索树的第K个节点

```python
def k_node(root, k):
    if not root or not k:
        return
    res = []

    def dfs(node):
        if len(res) >= k or not node:
            return
        dfs(node.left)
        res.append(node)
        dfs(node.right)
    dfs(root)
    if len(res) < k:
        return
    return res[k - 1]
```

#### 面试题55： 二叉树的深度

```python
# 二叉树深度


def tree_depth(root):
    if not root:
        return 0
    left = tree_depth(root.left)
    right = tree_depth(root.right)
    return max(left, right) + 1


# 平衡二叉树
def banlanced(root, depth):
    if not root:
        depth = 0
        return True
    left = right = 0
    if banlanced(root.left, left) and banlanced(root.right, right):
        diff = left - right
        if -1 <= diff <= 1:
            depth = left + 1 if left > right else right + 1
            return True
    return False


def is_banlanced(root):
    if not root:
        return
    depth = 0
    return banlanced(root, depth)
```

#### 面试题56： 数组中数字出现的次数

```python
def find_nums_appear_once(data):
    if not data or len(data) < 2:
        return
    res = 0
    for num in data:
        res ^= num
    index = first = second = 0
    while res & 1 == 0:
        res = res >> 1
        index += 1
    for num in data:
        if (num >> index) & 1:
            first ^= num
        else:
            second ^= num
    return (first, second)


# 数组中唯一只出现一次的数字

def find_number_appearing_once(data):
    if not data:
        return
    bitSum = [0] * 32
    for i in range(len(data)):
        bitMask = 1
        for j in reversed(range(32)):
            bit = data[i] & bitMask
            if bit:
                bitSum[j] += 1
            bitMask = bitMask << 1
    result = 0
    for i in range(32):
        result = result << 1
        result += bitSum[i] % 3
    return result
```

#### 面试题57： 和为s的数字

```python
def find_numbers_with_sum(data, num):
    if not data or not num:
        return False
    left = 0
    right = len(data) - 1
    while left < right:
        curSum = data[left] + data[right]
        if curSum == num:
            return data[left], data[right]
        elif curSum > num:
            right -= 1
        else:
            left -= 1
    return False


# 和为s的连续正数序列
def find_continuous_sequence(sum):
    if not sum or sum < 3:
        return
    small = 1
    big = 2
    middle = (1 + sum) >> 1
    curSum = small + big
    while small < middle:
        if curSum == sum:
            print_continuous_sequence(small, big)
        while curSum > sum and small < middle:
            curSum -= small
            small += 1
            if curSum == sum:
                print_continuous_sequence(small, big)
        big += 1
        curSum += big
```

#### 面试题58： 翻转字符串

```python
def reverse_string(data, start, end):
    while start < end:
        data[start], data[end] = data[end], data[start]
        start += 1
        end -= 1


def reverse_sentence(data):
    if not data:
        return
    start, end = 0, len(data) - 1
    data = list(data)
    reverse_string(data, start, end)

    start = end = 0
    while end < len(data):
        if data[end] == ' ' or end == len(data) - 1:
            ends = end if end == len(data) - 1 else end - 1
            reverse_string(data, start, ends)
            start = end = end + 1
        else:
            end += 1
    return ''.join(data)


# 左旋转字符串
def left_rotate_string(data, n):
    if not data:
        return
    if len(data) > 0 and n < len(data) and n > 0:
        firstStart = 0
        firstEnd = n - 1
        secondStart = n
        secondEnd = len(data) - 1

        data = list(data)
        reverse_string(data, firstStart, firstEnd)
        reverse_string(data, secondStart, secondEnd)
        reverse_string(data, firstStart, secondEnd)
        return ''.join(data)
```

#### 面试题59： 队列的最大值

```python
def max_in_windows(nums, size):
    if not size or size > len(nums):
        return []
    return [max(nums[i:i + size]) for i in range(len(nums) - size + 1)]
```

#### 面试题60： n个骰子的点数

```python
暂时无解
```

#### 面试题61： 扑克牌中的顺子

```python
def is_continuous(numbers):
    if not numbers:
        return False
    length = len(numbers)
    if length < 1:
        return False
    numbers = sorted(numbers)
    number_of_zero = len([x for x in numbers if x == 0])
    number_of_gap = 0
    small = number_of_zero
    big = small + 1
    while big < length:
        if numbers[small] == numbers[big]:
            return False
        number_of_gap += numbers[big] - numbers[small] - 1
        small = big
        big += 1
    return False if number_of_gap > number_of_zero else True
```

#### 面试题62： 圆圈中最后剩下的数字

```python
def last_remaining(n, m):
    if not n or not m:
        return -1
    last = 0
    for i in range(2, n + 1):
        last = (last + m) % i
    return last
```

#### 面试题63： 股票的最大利润

```python
def max_diff(numbers):
    if not numbers or len(numbers) < 2:
        return 0
    min = numbers[0]
    maxDiff = numbers[1] - min
    for i in range(2, len(numbers) - 1):
        if numbers[i] < min:
            min = numbers[i]
        curDiff = numbers[i] - min
        if curDiff > maxDiff:
            maxDiff = curDiff
    return maxDiff
```

#### 面试题64： 求1 + 2 +···+n

```python
def sum_solution(n):
    return sum(range(n + 1))
```

#### 面试题65： 不用加减乘除做加法

```python
def add_num(num1, num2):
    if not num1 or not num2:
        return
    return sum([num1, num2])
```

#### 面试题66： 构建乘积数组

```Python
def multiply(arrayA):
    head = [1]
    tail = [1]
    for i in range(len(arrayA) - 1):
        head.append(arrayA[i] * head[i])
        tail.append(arrayA[-i - 1] * tail[i])
    return [head[j] * tail[-j - 1] for j in range(len(head))]
```
