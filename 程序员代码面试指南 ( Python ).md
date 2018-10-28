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

    def josephusKill2(self, head, m):
        if not head or head.next == head or m < 1:
            return head
        cur = head.next
        tmp = 1
        while cur != head:
            tmp += 1
            cur = cur.next
        tmp = self.getLive(tmp, m)
        while tmp - 1 != 0:
            head = head.next
            tmp -= 1
        head.next = head
        return head

    def getLive(self, i, m):
        if i == 1:
            return 1
        return (self.getLive(i - 1, m) + m - 1) % i + 1


```

*****

判断一个链表是否为回文结构
```python


class LinkNode:
    def __init__(self, val, next=None):
        self.val = val
        self.next = next


class Solution:
    def isPalindrome1(self, head):
        if not head:
            return
        stack = []
        cur = head
        while cur:
            stack.append(cur)
            cur = cur.next
        while head:
            if head.val != stack.pop().val:
                return False
            head = head.next
        return True

    def isPalindrome2(self, head):
        if not head or not head.next:
            return
        stack = []
        cur = head
        right = head.next
        while cur.next and cur.next.next:
            cur = cur.next.next
            right = right.next
        while right:
            stack.append(right)
            right = right.next
        while stack:
            if head.val != stack.pop().val:
                return False
            head = head.next
        return True

    def isPalindrome3(self, head):
        if not head or not head.next:
            return
        n1 = n2 = head
        while n2.next and n2.next.next:
            n1, n2 = n1.next, n2.next.next
        n2 = n1.next
        n1.next = n3 = None
        while n2:
            n3 = n2.next
            n2.next = n1
            n1 = n2
            n2 = n3
        n2, n3 = head, n1
        res = True
        while n1 and n2:
            if n1.val != n2.val:
                return False
            n1, n2 = n1.next, n2.next
        n1 = n3.next
        n3.next = None
        while n1:
            n2 = n1.next
            n1.next = n3
            n3 = n1
            n1 = n2
        return res


```

*****

将单向链表按某值划分成左边小，中间相等，右边大的形式
```python


class LinkNode:
    def __init__(self, val, next=None):
        self.val = val
        self.next = next


class Solution:
    def listPartition1(self, head, pivot):
        if not head or not pivot:
            return head
        nodeArr = []
        cur = head
        while cur:
            nodeArr.append(cur)
            cur = cur.next
        self.arrPartition(nodeArr, pivot)
        for i in range(len(nodeArr) - 1):
            nodeArr[i].next = nodeArr[i + 1]
        nodeArr[-1].next = None
        return nodeArr[0]

    def arrPartition(self, nodeArr, pivot):
        small = 0
        big = len(nodeArr) - 1
        index = 0
        while index != big:
            if nodeArr[index].val < pivot:
                nodeArr[small], nodeArr[index] = nodeArr[index], nodeArr[small]
                small += 1
                index += 1
            elif nodeArr[index].val == pivot:
                index += 1
            else:
                nodeArr[big], nodeArr[index] = nodeArr[index], nodeArr[big]
                big -= 1

    def listPartition2(self, head, pivot):
        sH = sT = eH = eT = bH = bT = None
        while head:
            next = head.next
            head.next = None
            if head.val < pivot:
                if not sH:
                    sH = sT = head
                else:
                    sT.next = head
                    sT = head
            elif head.val == pivot:
                if not eH:
                    eH = eT = head
                else:
                    eT.next = head
                    eT = head
            else:
                if not bH:
                    bH = bT = head
                else:
                    bT.next = head
                    bT = head
            head = next

        if sT:
            sT.next = eH
            eT = eT if eT else sT

        if eT:
            eT.next = bH

        return sH if sH else eH if eH else bH


```

*****

两个单链表生成相加链表
```python
class LinkNode:
    def __init__(self, value, next=None):
        self.value = value
        self.next = next


class Solution:
    def addLists1(self, head1, head2):
        s1 = []
        s2 = []
        while head1:
            s1.append(s1.value)
            s1 = s1.next
        while head2:
            s2.append(s2.value)
            s2 = s2.next
        ca = 0
        n1 = 0
        n2 = 0
        n = 0
        node = None
        pre = None
        while s1 or s2:
            n1 = 0 if not s1 else s1.pop()
            n2 = 0 if not s2 else s2.pop()
            n = n1 + n2 + ca
            pre = node
            node = LinkNode(n % 10)
            node.next = pre
            ca = n // 10
        if ca == 1:
            pre = node
            node = LinkNode(1)
            node.next = pre
        return node

    def addLists2(self, head1, head2):
        head1 = self.reverseList(head1)
        head2 = self.reverseList(head2)
        ca = 0
        n1 = 0
        n2 = 0
        n = 0
        c1 = head1
        c2 = head2
        node = None
        pre = None
        while c1 or c2:
            n1 = c1.value if c1 else 0
            n2 = c2.value if c2 else 0
            n = n1 + n2 + ca
            pre = node
            node = LinkNode(n % 10)
            node.next = pre
            ca = n // 10
            c1 = c1.next if c1 else None
            c2 = c2.next if c2 else None
        if ca == 1:
            pre = node
            node = LinkNode(1)
            node.next = pre
        self.reverseList(head1)
        self.reverseList(head2)
        return node

    def reverseList(self, head):
        pre = None
        while head:
            next = head.next
            head.next = pre
            pre = head
            head = next
        return pre
```

*****

两个单链表相交的一系列问题
```python

class LinkNode:
    def __init__(self, value, next=None):
        self.value = value
        self.next = next

class Solution:

    def getIntersectNode(self, head1, head2):
        if not head1 or not head2:
            return None
        loop1 = self.getLoopNode(head1)
        loop2 = self.getLoopNode(head2)
        if not loop1 and not loop2:
            return self.noLoop(head1, head2)
        if loop1 and loop2:
            return self.bothLoop(head1, loop1, head2, loop2)
        return None

    def getLoopNode(self, head):
        if not head or not head.next or head.next.next:
            return None
        n1 = head.next
        n2 = head.next.next
        while n1 != n2:
            if not n2.next or not n2.next.next:
                n2 = n2.next.next
                n1 = n1.next

        n2 = head
        while n1 != n2:
            n1 = n1.next
            n2 = n2.next
        return n1

    def noLoop(self, head1, head2):
        if not head1 or not head2:
            return None
        cur1 = head1
        cur2 = head2
        n = 0
        while cur1.next:
            n += 1
            cur1 = cur1.next
        while cur2.next:
            n -= 1
            cur2 = cur2.next
        if cur1 != cur2:
            return None

        cur1 = head1 if n > 0 else head2
        cur2 = head2 if cur1 == head1 else head1
        n = abs(n)
        while n != 0:
            n -= 1
            cur1 = cur1.next
        while cur1 != cur2:
            cur1 = cur1.next
            cur2 = cur2.next
        return cur1

    def bothLoop(self, head1, loop1, head2, loop2):
        cur1 = None
        cue2 = None
        if loop1 == loop2:
            cur1 = head1
            cur2 = head2
            n = 0
            while cur1 != loop1:
                n += 1
                cur1 = cur1.next
            while cur2 != loop2:
                n -= 1
                cur2 = cur2.next
            cur1 = head1 if n > 0 else head2
            cur2 = head2 if cur1 == head1 else head1
            n = abs(n)
            while n != 0:
                n -= 1
                cur1 = cur1.next
            while cur1 != cur2:
                cur1 = cur1.next
                cur2 = cur2.next
            return cur1
        else:
            cur1 = loop1.next
            while cur1 != loop1:
                if cur1 == loop2:
                    return cur1
                cur1 = cur1.next
            return None
```

*****

将单链表的每K个节点之间逆序
```python

class LinkNode:
    def __init__(self, value, next=None):
        self.value = value
        self.next = next
        
class Solution:
    def reverseKnodes1(self, head, K):
        if K < 2:
            return head

        stack = []
        newHead = head
        cur = head
        pre = None
        next = None
        while cur:
            next = cur.next
            stack.append(cur)
            if len(stack) == K:
                pre = self.resign1(stack, pre, next)
                newHead = cur if newHead == head else newHead
            cur = next
        return newHead

    def resign1(self, stack, left, right):
        cur = stack.pop()
        if left:
            left.next = cur
        next = None
        while stack:
            next = stack.pop()
            cur.next = next
            cur = next
        cur.next = right
        return cur

    def reverseKNodes2(self, head, K):
        if K < 2:
            return head

        cur = head
        start = None
        pre = None
        next = None
        count = 1
        while cur:
            next = cur.next
            if count == K:
                start = head if not pre else pre.next
                head = cur if not pre else head
                self.resign2(pre, start, cur, next)
                pre = start
                count = 0
            count += 1
            cur = next
        return head

    def resign2(self, left, start, end, right):
        pre = start
        cur = start.next
        next = None
        while cur != right:
            next = cur.next
            cur.next = pre
            pre = cur
            cur = next
        if left:
            left.next = end
        start.next = right
```

*****

删除无序单链表中值重复出现的节点
```python

class LinkNode:
    def __init__(self, value, next=None):
        self.value = value
        self.next = next
        
class Solution:
    def removeRep1(self, head):
        if not head:
            return
        valueSet = set()
        pre = head
        cur = head.next
        valueSet.add(pre.value)
        while cur:
            if cur.value in valueSet:
                pre.next = cur.next
            else:
                valueSet.add(cur.value)
                pre = cur
            cur = cur.next

    def removeRep2(self, head):
        cur = head
        pre = None
        next = None
        while cur:
            pre = cur
            next = cur.next
            while next:
                if cur.value == next.value:
                    pre.next = next.next
                else:
                    pre = next
                next = next.next
            cur = cur.next
```

*****

在单链表中删除指定值的节点

```python
class LinkNode:
    def __init__(self, value, next=None):
        self.value = value
        self.next = next
        

class Solution:
    def removeValue1(self, head, num):
        stack = []
        while head:
            if head.value != num:
                stack.append(head)
            head = head.next

        while stack:
            stack[-1].next = head
            head = stack.pop()

        return head

    def removeValue2(self, head, num):
        while head:
            if head.value == num:
                break
            head = head.next

        pre = head
        cur = head

        while cur:
            if cur.value == num:
                pre.next = cur.next
            else:
                pre = cur
            cur = cur.next
        return head
```

*****

将搜索二叉树转换为双向链表


```python
class DoubleLinkNode:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right


class Solution:
    def convert1(self, head):
        queue = []
        self.inOrderToQueue(head, queue)
        if not queue:
            return head
        head = queue.pop(0)
        pre = head
        pre.left = None
        cur = None
        while queue:
            cur = queue.pop(0)
            pre.right = cur
            cur.left = pre
            pre = cur
        pre.right = None
        return head

    def inOrderToQueue(self, head, queue):
        if not head:
            return
        self.inOrderToQueue(head.left, queue)
        queue.append(head)
        self.inOrderToQueue(head.right, queue)

    def convert2(self, head):
        if not head:
            return None
        last = self.process(head)
        head = last.right
        last.right = None
        return head
    
    def process(self, head):
        if not head:
            return None
        
        leftE = self.process(head.left)
        rightE = self.process(head.right)
        leftS = leftE.right if leftE else None
        rightS = rightE.right if rightE else None
        if leftE and rightE:
            leftE.right = head
            head.left = leftE
            head.right = rightS
            rightS.left = head
            rightE.right = leftS
            return rightE
        elif leftE:
            leftE.right = head
            head.left = leftE
            head.right = leftS
            return head
        elif rightE:
            head.right = rightS
            rightS.left = head
            rightE.right = head
            return rightE
        else:
            head.right = head
            return head

```

*****

单链表的选择排序

```python
class LinkNode:
    def __init__(self, value, next=None):
        self.value = value
        self.next = next


class Solution:
    def selectionSort(self, head):
        tail = None
        cur = head
        smallPre = None
        small = None
        while cur:
            small = cur
            smallPre = self.getSmallestPreNode(head)
            if smallPre:
                small = smallPre.next
                smallPre.next = small.next
            cur = cur.next if cur == small else cur
            if not tail:
                head = small
            else:
                tail.next = small
            tail = small
        return head

    def getSmallestPreNode(self, head):
        small = head
        smallPre = None
        pre = head
        cur = head.next
        while cur:
            if cur.value < small.value:
                small = cur
                smallPre = pre
            pre = cur
            cur = cur.next
        return smallPre

```

*****

一种怪异的节点删除方式

```python
class Solution:
    def removeNodeWired(self, node):
        if not node:
            return
        next = node.next
        if not next:
            raise Exception('Can not remove last node.')
        node.value = next.value
        node.next = next.next

```

*****

向有序的环形单链表中插入新节点
```python

class Solution:
    def insertNum(self, head, num):
        node = LinkNode(num)
        if not head:
            node.next = node
            return node

        pre = head
        cur = head.next
        while cur != head:
            if pre.value <= num <= cur.value:
                break
            pre = cur
            cur = cur.next

        pre.next = node
        node.next = cur
        return head if head.value < num else node
```

*****

合并两个有序的单链表

```python

class Solution:
    def merge(self, head1, head2):
        if not head1 or not head2:
            return head1 if head1 else head2
        head = head1 if head1.value < head2.value else head2
        cur1 = head1 if head == head1 else head2
        cur2 = head2 if head == head2 else head1
        pre = None
        next = None
        while cur1 and cur2:
            if cur1.value <= cur2.value:
                pre = cur1
                cur1 = cur1.next
            else:
                next = cur2.next
                pre.next = cur2
                cur2.next = cur1
                pre = cur2
                cur2 = next
        pre.next = cur2 if not cur1 else cur1
        return head
```

*****

按照左右半区的方式重新组合单链表

```python

class Solution:
    def relocate(self, head):
        if not head or not head.next:
            return

        mid = head
        right = head.next
        while right.next and right.next.next:
            mid = mid.next
            right = right.next.next
        right = mid.next
        mid.next = None
        self.mergeLR(head, right)

    def mergeLR(self, left, right):
        next = None
        while left.next:
            next = right.next
            right.next = left.next
            left.next = right
            left = right.next
            right = next
        left.next = right
```


 # 第3章 二叉树问题
 
分别用递归和非递归方式实现二叉树先序、中序和后序遍历

```python

class TreeNode:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right


class Solution:
    def preOrderRecur(self, head):
        if not head:
            return
        print(head.value, end=" ")
        self.preOrderRecur(head.left)
        self.preOrderRecur(head.right)

    def inOrderRecur(self, head):
        if not head:
            return
        self.inOrderRecur(head.left)
        print(head.value, end=" ")
        self.inOrderRecur(head.right)

    def posOrderRecur(self, head):
        if not head:
            return
        self.posOrderRecur(head.left)
        self.posOrderRecur(head.right)
        print(head.value, end=" ")

    def preOrderUnRecur(self, head):
        if head:
            stack = []
            stack.append(head)
            while stack:
                head = stack.pop()
                print(head.value, end=" ")
                if head.right:
                    stack.append(head.right)
                if head.left:
                    stack.append(head.left)

    def inOrderUnRecur(self, head):
        if head:
            stack = []
            while stack or head:
                if head:
                    stack.append(head)
                    head = head.left
                else:
                    head = stack.pop()
                    print(head.value, end=" ")
                    head = head.right

    def posOrderUnRecur1(self, head):
        if head:
            s1 = []
            s2 = []
            s1.append(head)
            while s1:
                head = s1.pop()
                s1.append(head)
                if head.left:
                    s1.append(head.left)
                if head.right:
                    s1.append(head.right)
            while s2:
                print(s2.pop(), end=" ")

    def posOrderUnRecur2(self, head):
        if head:
            stack = [head]
            c = None
            while stack:
                c = stack[-1]
                if c.left and head != c.left and head != c.right:
                    stack.append(c.left)
                elif c.right and head != c.right:
                    stack.append(c.right)
                else:
                    print(stack.pop(), end=" ")
                    head = c
```

*****

打印二叉树的边界节点

```python


```