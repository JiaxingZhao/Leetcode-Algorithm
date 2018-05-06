# runtime: 96ms
class Solution:
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        length = len(nums)
        if length <= 1:
            return length
        for i in reversed(range(1, length)):
            if nums[i] == nums[i-1]:
                nums.pop(i)
        return len(nums)
