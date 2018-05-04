# 第1章 栈和队列

设计一个有getMin功能的栈
```python
# 使用两个栈，一个栈用来保存当前栈中的元素，功能和正常的栈没有区别，记为stackData
# 另一个栈用来保存每一步的最小值，记为stackMin


class MyStack:
    def __init__(self):
        self.stackData = []
        self.stackMin = []

    def push(self, newNum):
        if not self.stackMin:
            self.stackMin.append(newNum)
        elif newNum <= self.stackMin[-1]:
            self.stackMin.append(newNum)
        self.stackData.append(newNum)

    def pop(self):
        if not self.stackData:
            print('Stack is Empty')
            return
        value = self.stackData.pop()
        if value == self.stackMin[-1]:
            self.stackMin.pop()
        return value

    def getmin(self):
        if not self.stackMin:
            print('Stack is Empty')
            return
        return self.stackMin[-1]


```

*****

由两个栈组成的队列
```python
# 栈的特点是先进后出，队列的特点是先进先出，使用两个栈正好把顺序反过来实现队列


class TwoStacksQueue:
    def __init__(self):
        self.stackPush = []
        self.stackPop = []

    def add(self, pushInt):
        self.stackPush.append(pushInt)

    def poll(self):
        if not self.stackPush and not self.stackPop:
            print('Queue is empty')
            return
        elif not self.stackPop:
            while self.stackPush:
                self.stackPop.append(self.stackPush.pop())
        return self.stackPop.pop()

    def peek(self):
        if not self.stackPush and not self.stackPop:
            print('Queue is empty')
            return
        elif not self.stackPop:
            while self.stackPush:
                self.stackPop.append(self.stackPush.pop())
        return self.stackPop[-1]


```

*****

如何仅用递归函数和栈操作逆序一个栈

```python
# 使用列表模拟栈
# 设计两个递归函数，第一个将栈stack的栈底元素返回并删除
# 第二个函数使用第一个递归函数逆序一个栈
# 每次递归会获得栈底元素为result，倒数第二个元素为last
# 获取到result之后删除栈底元素，然后将last重新压入栈中，当栈为空时返回
# 利用系统会保存每次递归的变量，在递归向上返回时重新压入栈中


class Solution:
    def getAndRemoveLastElement(self, stack):
        result = stack.pop()
        if not stack:
            return result
        else:
            last = self.getAndRemoveLastElement(stack)
            stack.append(result)
            return last

    def reverse_stack(self, stack):
        if not stack:
            return
        i = self.getAndRemoveLastElement(stack)
        self.reverse_stack(stack)
        stack.append(i)


```

*****

猫狗队列
```python


class PetEnterQueue:
    def __init__(self, pet: str, count: int):
        self.pet = pet
        self.count = count

    def getPet(self):
        return self.pet

    def getCount(self):
        return self.count


class DogCatQueue:
    def __init__(self):
        self.dogQ = []
        self.catQ = []
        self.count = 0

    def add_pet(self, pet: str):
        if pet.lower() == 'dog':
            self.count += 1
            self.dogQ.append(PetEnterQueue(pet, self.count))
        elif pet.lower() == 'cat':
            self.count += 1
            self.catQ.append(PetEnterQueue(pet, self.count))
        else:
            print('errot: not dog or cat')
            return

    def pollAll(self):
        if self.dogQ and self.catQ:
            if self.dogQ[0].getCount() < self.catQ[0].getCount():
                return self.dogQ.pop(0).getPet()
            else:
                return self.catQ.pop(0).getPet()
        elif self.dogQ:
            return self.dogQ.pop(0).getPet()
        elif self.catQ:
            return self.catQ.pop(0).getPet()
        else:
            print('error: Queue is empty')
            return

    def pollDog(self):
        if self.dogQ:
            return self.dogQ.pop(0).getPet()
        else:
            print('error: Dog queue is empty')

    def pollCat(self):
        if self.catQ:
            return self.catQ.pop(0).getPet()
        else:
            print('error: Cat queue is empty')

    def isEmpty(self):
        return not self.dogQ and not self.catQ

    def isDogQueueEmpty(self):
        return not self.dogQ

    def isCatQueueEmpty(self):
        return not self.catQ


```

*****

用一个栈实现另一个栈的排序
```python
# stack为要排序的栈， 申请辅助栈为help， 在stack执行pop操作， 弹出的标记记为cur


class Solution:
    def sortStackByStack(self, stack):
        if not stack:
            return
        help = []
        while stack:
            cur = stack.pop()
            while help and help[-1] < cur:
                stack.append(help.pop())
            help.append(cur)
        while help:
            stack.append(help.pop())


```

*****

用栈来求解汉诺塔问题
```python
# 递归版本


class HanoiProblem1:
    def hanoiProblem(self, num: int, left: str, mid: str, right: str):
        if num < 1:
            return 0
        return self.process(num, left, mid, right, left, right)

    def process(self, num: int, left: str, mid: str, right: str, fromP: str, toP: str):
        if num == 1:
            if fromP == 'mid' or toP == 'mid':
                print('Move 1 from {} to {}'.format(fromP, toP))
                return 1
            else:
                print('Move 1 from {} to {}'.format(fromP, mid))
                print('Move 1 from {} to {}'.format(mid, toP))
                return 2
        if fromP == 'mid' or toP == 'mid':
            another = 'right' if fromP == 'left' or toP == 'left' else 'left'
            part1 = self.process(num - 1, left, mid, right, fromP, another)
            part2 = 1
            print('Move {} from {} to {}'.format(num, fromP, toP))
            part3 = self.process(num - 1, left, mid, right, another, toP)
            return part1 + part2 + part3
        else:
            part1 = self.process(num - 1, left, mid, right, fromP, toP)
            part2 = 1
            print('Move {} from {} to {}'.format(num, fromP, mid))
            part3 = self.process(num - 1, left, mid, right, toP, fromP)
            part4 = 1
            print('Move {} from {} to {}'.format(num, mid, toP))
            part5 = self.process(num - 1, left, mid, right, fromP, toP)
            return part1 + part2 + part3 + part4 + part5


# 非递归版本 -- 用栈来模拟整个过程


class HanoiProblem2:
    def hanoiProblem(self, num: int, left: str, mid: str, right: str):
        LS = [2**31]
        MS = [2**31]
        RS = [2**31]
        for i in range(num, 0, -1):
            LS.append(i)
        step = 0
        record = ''
        while len(RS) != num + 1:
            record, step = self.fStackTotStack(record, 'MToL', 'LToM', LS, MS, 'left', 'mid')
            step += step
            record, step = self.fStackTotStack(record, 'LToM', 'MToL', MS, LS, 'mid', 'left')
            step += step
            record, step = self.fStackTotStack(record, 'RToM', 'MToR', MS, RS, 'mid', 'right')
            step += step
            record, step = self.fStackTotStack(record, 'MToR', 'RToM', RS, MS, 'right', 'mid')
            step += step
        return step

    def fStackTotStack(self, record, preAction, nowAction, fStack, tStack, sFrom, sTo):
        if record != preAction and fStack[-1] < tStack[-1]:
            tStack.append(fStack.pop())
            print('Move {} from {} to {}'.format(tStack[-1], sFrom, sTo))
            record = nowAction
            return record, 1
        else:
            return record, 0


```

*****

生成窗口最大值数组
```python
# Pythonic 版本


class Solution:
    def getMaxWindow(self, arr, w):
        return [max(arr[i:i + w]) for i in range(len(arr) - w + 1)]


```

*****

构造数组的MaxTree
```python


class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def getMaxtree(self, arr: list):
        narr = [0] * len(arr)
        for i in range(len(arr)):
            narr[i] = Node(arr[i])
        stack = []
        LBigMap = {}
        RBigMap = {}
        for i in range(len(arr)):
            curNode = narr[i]
            while stack and stack[-1].val < curNode.val:
                self.popStackSetMap(stack, LBigMap)
            stack.append(curNode)
        while stack:
            self.popStackSetMap(stack, LBigMap)
        for i in range(len(arr) - 1, -1, -1):
            curNode = narr[i]
            while stack and stack[-1].val < curNode.val:
                self.popStackSetMap(stack, RBigMap)
            stack.append(curNode)
        while stack:
            self.popStackSetMap(stack, RBigMap)
        head = None
        for i in range(len(narr)):
            curNode = narr[i]
            left = LBigMap.get(curNode)
            right = RBigMap.get(curNode)
            if not left and not right:
                head = curNode
            elif not left:
                if not right.left:
                    right.left = curNode
                else:
                    right.right = curNode
            elif not right:
                if not left.left:
                    left.left = curNode
                else:
                    left.right = curNode
            else:
                parent = left if left.val < right.val else right
                if not parent.left:
                    parent.left = curNode
                else:
                    parent.right = curNode
        return head

    def popStackSetMap(self, stack, map):
        popNode = stack.pop()
        map[popNode] = None if not stack else stack[-1]


```

*****

求最大子矩阵的大小
```python


class Solution:
    def maxRecSize(self, map):
        if not map or len(map[0]) == 0:
            return 0
        maxArea = 0
        height = [0] * len(map[0])
        for i in range(len(map)):
            for j in range(len(map[0])):
                height[j] = 0 if map[i][j] == 0 else height[j] + 1
            maxArea = max(self.maxRecFromBottom(height), maxArea)
        return maxArea

    def maxRecFromBottom(self, height):
        if not height:
            return 0
        maxArea = 0
        stack = []
        for i in range(len(height)):
            while stack and height[i] <= height[stack - 1]:
                j = stack.pop()
                k = -1 if not stack else stack[-1]
                curArea = (i - k - 1) * height[j]
                maxArea = max(maxArea, curArea)
                stack.append(i)
            while stack:
                j = stack.pop()
                k = -1 if not stack else stack[-1]
                curArea = (len(height) - k - 1) * height[j]
                maxArea = max(maxArea, curArea)
            return maxArea


```

*****

最大值减去最小值小于或等于num的子数组数量
```python
# 设置两个辅助栈，分别保存当前找到的最大值和最小值，找到后即对比是否满足num要求
# 如果不满足则跳过当前循环，继续判断当前指针是否大于栈最后一个值，满足则弹出最后一个值
# 当前i指针组成的所有子数组满足条件数量的数量为j，以arr[i+1]作为第一个元素子数组，结果res+=j-i


class Solution:
    def getNum(self, arr, num):
        if not arr:
            return 0
        qmin = []
        qmax = []
        i = 0
        j = 0
        res = 0
        while i < len(arr):
            while j < len(arr):
                while qmin and arr[qmin[-1]] >= arr[j]:
                    qmin.pop()
                qmin.append(j)
                while qmax and arr[qmax[-1]] <= arr[j]:
                    qmax.pop()
                qmax.append(j)
                if (arr[qmax[0]] - arr[qmin[0]]) > num:
                    break
                j += 1
            if qmin[0] == i:
                qmin.pop(0)
            if qmax[0] == i:
                qmax.pop(0)
            res += j - i
            i += 1
        return res


```


# 第2章 链表问题

打印两个有序链表的公共部分
```python


class LinkNode:
    def __init__(self, val, next=None):
        self.val = val
        self.next = next


class Solution:
    def printCommonPart(self, head1, head2):
        if not head1 or not head2:
            return
        while head1 and head2:
            if head1.val < head2.val:
                head1 = head1.next
            elif head2.val < head1.val:
                head2 = head2.next
            else:
                print(head1.val, end=' ')
                head1 = head1.next
                head2 = head2.next


```

*****

在单链表和双链表中删除倒数第K个节点
```python


class LinkNode:
    def __init__(self, val, next=None):
        self.val = val
        self.next = next


class DoubleNode:
    def __init__(self, val, last=None, next=None):
        self.val = val
        self.last = last
        self.next = next


class Solution:
    def removeLastKthNodeInSingleLink(self, head, lastKth):
        if not head or lastKth < 1:
            return head
        cur = head
        while cur:
            lastKth -= 1
            cur = cur.next
        if lastKth == 0:
            head = head.next
        if lastKth < 0:
            cur = head
            while lastKth != 0:
                cur = cur.next
                lastKth += 1
            cur.next = cur.next.next
        return head

    def removeLastKthNodeInDoubleLink(self, head, lastKth):
        if not head or lastKth < 1:
            return
        cur = head
        while cur:
            lastKth -= 1
            cur = cur.next
        if lastKth == 0:
            head = head.next
            head.last = None
        if lastKth < 0:
            cur = head
            while lastKth != 0:
                cur = cur.next
                lastKth += 1
            newNext = cur.next.next
            cur.next = newNext
            if newNext:
                newNext.last = cur
        return head


```

*****

删除链表的中间节点和a / b处的节点
```python


class LinkNode:
    def __init__(self, val, next=None):
        self.val = val
        self.next = next


class Solution:
    def removeMidNode(self, head):
        if not head or not head.next:
            return head
        pre = head
        cur = head.next.next
        while cur.next and cur.next.next:
            pre = pre.next
            cur = cur.next.next
        pre.next = pre.next.next
        return head

    def removeByRatio(self, head, a, b):
        if a < 1 or a > b:
            return head
        n = 0
        cur = head
        while cur:
            n += 1
            cur = cur.next
        import math
        n = math.ceil((a * n) / b)
        if n == 1:
            head = head.next
        if n > 1:
            cur = head
            while n - 1 != 1:
                cur = cur.next
            cur.next = cur.next.next
        return head


```

*****

反转单向和双向链表
```python


class LinkNode:
    def __init__(self, val, next=None):
        self.val = val
        self.next = next


class DoubleLinkNode:
    def __init__(self, val, last=None, next=None):
        self.val = val
        self.last = last
        self.next = next


class Solution:
    def reverseSingleLink(self, head):
        pre = None
        next = None
        while head:
            next = head.next
            head.next = pre
            pre = head
            head = next
        return pre

    def reverseDoubleLink(self, head):
        pre = None
        next = None
        while head:
            next = head.next
            head.next = pre
            head.last = next
            pre = head
            head = next
        return pre


```

*****

反转部分单向链表
```python


class LinkNode:
    def __init__(self, val, next=None):
        self.val = val
        self.next = next


class Solution:
    def reversePart(self, head, fromN, toN):
        node1 = head
        fPre = None
        tPos = None
        length = 0
        while node1:
            length += 1
            fPre = node1 if length == fromN - 1 else fPre
            tPos = node1 if length == toN + 1 else tPos
            node1 = node1.next
        if fromN > toN or fromN < 1 or toN > length:
            return head
        node1 = head if not fPre else fPre.next
        node2 = node1.next
        node1.next = tPos
        next = None
        while node2 != tPos:
            next = node2.next
            node2.next = node1
            node1 = node2
            node2 = next
        if fPre:
            fPre.next = node1
            return head
        return node1


```

*****

形成单链表的约瑟夫问题
```python


class Solution:
    def josephusKill1(self, head, m):
        if not head or head == head.next or m < 1:
            return head
        last = head
        while last.next != head:
            last = last.next
        count = 0
        while head != last:
            if count + 1 == m:
                last.next = head.next
                count = 0
            else:
                last = last.next
            head = last.next
            count += 1
        return head


```
