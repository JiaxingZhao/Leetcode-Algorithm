------



------

### #13  Roman to Integer  --  Easy

Given a roman numeral, convert it to an integer.

Input is guaranteed to be within the range from 1 to 3999.

**Solution:**

```python
class Solution:
    def romanToInt(self, s):
        total = 0
        roman = {'M': 1000,'D': 500 ,'C': 100,'L': 50,'X': 10,'V': 5,'I': 1}
        length = len(s)
        for i in range(length):
            if i == length-1 :
                total += roman[s[i]]
            else:
                total = (total + roman[s[i]]) if roman[s[i]] >= roman[s[i+1]] else (total - roman[s[i]])
        return total
    	# 遍历每一个罗马数字，如果是最后一位 不做判断，直接相加，其他位置判断是否大于下一位，大于则加，小于则减，例如 XI == 11 / X == 9 / IX == 9
        # Runtime: 140 ms
```



------

### #344  Reverse String  --  Easy

Write a function that takes a string as input and returns the string reversed.

**Example:**

```
Given s = "hello", return "olleh".
```

**Solution:**

```python
class Solution:
    def reverseString(self, s):
        return s[::-1]
   		# 直接返回字符串倒序
        # Runtime: 52 ms
```



------

### #383  Ransom Note  --  Easy

Given an arbitrary ransom note string and another string containing letters from all the magazines, write a function that will return true if the ransom note can be constructed from the magazines ; otherwise, it will return false.

Each letter in the magazine string can only be used once in your ransom note.

**Note:**
You may assume that both strings contain only lowercase letters.

```
canConstruct("a", "b") -> false
canConstruct("aa", "ab") -> false
canConstruct("aa", "aab") -> true
```

**Solution：**

```python
class Solution:
    def canConstruct(self, ransomNote, magazine):
        ransom = collections.Counter(ransomNote)
        maga = collections.Counter(magazine)
        return not ransom - maga
    	# 使用collections的Counter统计字符串中的字符数量，然后计算差异值
        # Runtime: 64 ms
```



------

### #520  Detect Capital  --  Easy

Given a word, you need to judge whether the usage of capitals in it is right or not.

We define the usage of capitals in a word to be right when one of the following cases holds:

1. All letters in this word are capitals, like "USA".
2. All letters in this word are not capitals, like "leetcode".
3. Only the first letter in this word is capital if it has more than one letter, like "Google".

Otherwise, we define that this word doesn't use capitals in a right way.

**Example 1:**

```
Input: "USA"
Output: True
```

**Example 2:**

```
Input: "FlaG"
Output: False
```

**Note:** The input will be a non-empty word consisting of uppercase and lowercase latin letters.

**Solution:**

```python
class Solution:
    def detectCapitalUse(self, word):
        return word.isupper() or word.islower() or (word[0].isupper() and word[1:].islower()) 
    	# 对word进行大小写判定，直接返回结果
        # Runtime: 52 ms
```



------

### #521  Longest Uncommon Subsequence I  --  Easy

Given a group of two strings, you need to find the longest uncommon subsequence of this group of two strings. The longest uncommon subsequence is defined as the longest subsequence of one of these strings and this subsequence should not be **any **subsequence of the other strings.

A **subsequence** is a sequence that can be derived from one sequence by deleting some characters without changing the order of the remaining elements. Trivially, any string is a subsequence of itself and an empty string is a subsequence of any string.

The input will be two strings, and the output needs to be the length of the longest uncommon subsequence. If the longest uncommon subsequence doesn't exist, return -1.

**Example 1:**

```
Input: "aba", "cdc"
Output: 3
Explanation: The longest uncommon subsequence is "aba" (or "cdc"), 
because "aba" is a subsequence of "aba", 
but not a subsequence of any other strings in the group of two strings. 
```

**Note:**

1. Both strings' lengths will not exceed 100.
2. Only letters from a ~ z will appear in input strings.

**Solution:**

```python
class Solution:
    def findLUSlength(self, a, b):
        return -1 if a == b else max(len(a), len(b))
    	# 要求A!=B ,如果等于直接返回-1，否则判断max
        # Runtime: 40 ms
```



------

### #551  Student Attendance Record I  --  Easy

You are given a string representing an attendance record for a student. The record only contains the following three characters:

1. **'A'** : Absent.
2. **'L'** : Late.
3. **'P'** : Present.

A student could be rewarded if his attendance record doesn't contain **more than one 'A' (absent)** or **more than two continuous 'L' (late)**.

You need to return whether the student could be rewarded according to his attendance record.

**Example 1:**

```
Input: "PPALLP"
Output: True
```

**Example 2:**

```
Input: "PPALLL"
Output: False
```

**Solution：**

```python
class Solution:
    def checkRecord(self, s):
        return s.count("A") <= 1 and "LLL" not in s
    	# 同时满足条件不超过一个A或有连续三个或以上L返回True
        #Runtime: 52 ms
```



------

### #557  Reverse Words in a String III  --  Easy

Given a string, you need to reverse the order of characters in each word within a sentence while still preserving whitespace and initial word order.

**Example 1:**

```
Input: "Let's take LeetCode contest"
Output: "s'teL ekat edoCteeL tsetnoc"
```

**Note:** In the string, each word is separated by single space and there will not be any extra space in the string.

**Solution:**

```python
class Solution:
    def reverseWords(self, s):
        return " ".join(map(lambda x: x[::-1], s.split()))
    	# 根据空格分隔单词，分隔后单词倒序，然后再用空格拼接
        # Runtime: 40 ms
```



------

### #657  Judge Route Circle  --  Easy

Initially, there is a Robot at position (0, 0). Given a sequence of its moves, judge if this robot makes a circle, which means it moves back to **the original place**.

The move sequence is represented by a string. And each move is represent by a character. The valid robot moves are `R` (Right), `L`(Left), `U` (Up) and `D` (down). The output should be true or false representing whether the robot makes a circle.

**Example 1:**

```
Input: "UD"
Output: true
```

**Example 2:**

```
Input: "LL"
Output: false
```

**Solution:**

```python
class Solution:
    def judgeCircle(self, moves):
        return moves.count("U") == moves.count("D") and moves.count("L") == moves.count("R")
		# 判断UD数量和LR数量，因为要形成循环，所以如果数量不相等，则肯定不是循环
```



------

### #788  Rotated Digits  --  Easy

X is a good number if after rotating each digit individually by 180 degrees, we get a valid number that is different from X. A number is valid if each digit remains a digit after rotation. 0, 1, and 8 rotate to themselves; 2 and 5 rotate to each other; 6 and 9 rotate to each other, and the rest of the numbers do not rotate to any other number.

Now given a positive number `N`, how many numbers X from `1` to `N` are good?

```
Example:
Input: 10
Output: 4
Explanation: 
There are four good numbers in the range [1, 10] : 2, 5, 6, 9.
Note that 1 and 10 are not good numbers, since they remain unchanged after rotating.
```

**Note:**

- N  will be in range `[1, 10000]`.

**Solution:**

```python
class Solution:
    def rotatedDigits(self, N):
        count = 0
        for i in range(1,N+1):
            i = str(i)
            if any(x in i for x in ["3","4","7"]):
                continue
            if any(x in i for x in ["2","5","6","9"]):
                count += 1
        return count
    	# forin所有N，符合条件相加，其他continue
        # Runtime: 140 ms
```



------

