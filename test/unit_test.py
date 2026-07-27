#!/user/bin/env python
# encoding=utf-8
# @Author: daping
# @Date: 2024/7/31 13:00
import random
import unittest


class TestMethods(unittest.TestCase):

    def test_1(selfs):
        print("unittest")
        import random

        def generate_random_chinese_char():
            # Unicode 范围：常用汉字的范围是 0x4E00 到 0x9FA5
            return chr(random.randint(0x4E00, 0x9FA5))

        def generate_random_chinese_line(length=10):
            return ''.join(generate_random_chinese_char() for _ in range(length))

        def generate_random_chinese_text(lines=100, line_length=10):
            return [generate_random_chinese_line(line_length) for _ in range(lines)]


        random_chinese_text = generate_random_chinese_text()
        for line in random_chinese_text:
            print(line)


    def test_2(selfs):
        random.uniform

        def locked_dice_game(total_rounds, max_score):
            """
            :param total_rounds: 游戏总轮数 N (本题为 100)
            :param max_score: 骰子最大分数 M (本题为 20)
            :return: 最优策略下的总期望分数
            """
            # 1. 边界条件：如果只剩最后 1 轮，无论掷出多少都必须锁定
            # 期望收益就是 1~20 的平均值
            expected = sum(range(1, max_score + 1)) / max_score

            # 2. 从倒数第 2 轮开始，向前逆推直到第 1 轮
            # remaining_rounds 代表当前这一轮加上后面所有的轮数
            for remaining_rounds in range(2, total_rounds + 1):
                lock_sum = 0  # 记录选择“锁定”的点数总和
                continue_count = 0  # 记录选择“放弃”的点数个数

                # 遍历所有可能的点数 (1 到 20)
                for score in range(1, max_score + 1):
                    # 比较：锁定收益 vs 放弃收益
                    if score * remaining_rounds > expected:
                        lock_sum += score
                    else:
                        continue_count += 1

                print(expected, lock_sum, continue_count, expected/remaining_rounds)

                # 3. 计算当前轮的期望收益
                # (锁定选择的总分 * 剩余轮数 + 放弃选择的期望 * 放弃的次数) / 总可能数
                expected = (lock_sum * remaining_rounds + continue_count * expected) / max_score
                print(expected)

            return expected

        # 测试题目参数：N=100，M=20
        result = locked_dice_game(100, 20)
        print(f"最优策略下的总期望分数为: {result:.4f}")

    def test_2(self):
        def minDistance(word1: str, word2: str) -> int:

            m = len(word1)
            n = len(word2)

            dp = [[0] * (n + 1) for _ in range(m + 1)]

            for i in range(m + 1):
                dp[i][0] = i

            for j in range(n + 1):
                dp[0][j] = j

            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if word1[i - 1] == word2[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1]
                    else:
                        dp[i][j] = min(dp[i - 1][j - 1], dp[i][j - 1], dp[i - 1][j]) + 1

            return dp[m][n]
        print(minDistance("delete", "leet"))

    def test_3(self):
        import heapq
        nums = [3,2,1,5,6,4]
        k = 2
        min_heap = []  # 初始化一个小顶堆
        for num in nums:
            # 将当前元素压入堆中
            heapq.heappush(min_heap, num)
            # 如果堆的容量超过了 k，弹出堆顶元素（即当前堆中的最小值）
            # 这样可以保证堆里始终只保留“目前遇到的最大的 k 个数”
            if len(min_heap) > k:
                heapq.heappop(min_heap)
        # 遍历结束后，堆里剩下的就是数组中最大的 k 个数
        # 因为是小顶堆，堆顶（索引 0）就是这 k 个数中最小的，也就是第 k 大的元素
        print(min_heap[0])


    def test_4(self):
        def plusOne(digits, k):

            n = len(digits)

            pre = (digits[n - 1] + k) // 10 if digits[n - 1] + k >= 10 else 0
            digits[n - 1] = (digits[n - 1] + k) % 10

            #print(digits, pre)

            for i in range(n - 2, -1, -1):
                # print(digits, pre)
                if digits[i] + pre >= 10:
                    digits[i] = (digits[i] + pre) % 10
                    pre = 1
                else:
                    digits[i] = digits[i] + pre
                    pre = 0
                # print(digits)

            # print(digits, pre)
            if pre == 0:
                return digits
            else:
                return [1] + digits

        print(plusOne([5], 5))