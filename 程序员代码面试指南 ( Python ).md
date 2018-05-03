### 第1章 栈和队列

设计一个有getMin功能的栈
```python
# 使用两个栈，一个栈用来保存当前栈中的元素，功能和正常的栈没有区别，记为stackData
# 另一个栈用来保存每一步的最小值，记为stackMin

class Solution:
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

```