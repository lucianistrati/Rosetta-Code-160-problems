"""
python concepts

Map, Filter, Reduce
subprocess
threading
finate state machine: fysom library
__name__
Iterable and Iterator
yield Keyword
Python Generators
Python Closures
Python Decorators
@property Decorator in Python
Assert Statement
Garbage Collection


https://automatetheboringstuff.com/

cool book
"""
# PROBLEMS: https://www.freecodecamp.org/learn/coding-interview-prep/

# 1
# def diff(a, b):
#     return set(a).difference(set(b))
#
#
# def sym(a, b):
#     return list(diff(a, b).union(diff(b, a)))
#
#
# print(sym([1, 2, 3], [5, 2, 1, 4]))

# 2
# def lookup(cur_inv, search_name_):
#     for i, elem in enumerate(cur_inv):
#         qnt, name = elem
#         if name == search_name_:
#             return i
#     return -1
#
#
# def inventory_update(cur_inv, new_inv):
#     for elem in new_inv:
#         qnt, name = elem
#         idx = lookup(cur_inv, name)
#         if idx == -1:
#             cur_inv.append(elem)
#         else:
#             cur_inv[idx][0] += qnt
#     return cur_inv
#
# cur_inv = [[21, "Bowling Ball"], [2, "Dirty Sock"], [1, "Hair Pin"], [5, "Microphone"]]
# new_inv = [[2, "Hair Pin"], [3, "Half-Eaten Apple"], [67, "Bowling Ball"], [7, "Toothpaste"]]
#
#
# res = inventory_update(cur_inv, new_inv)
# print(res)


# 3
# from itertools import permutations
#
# def is_good(a: str):
#     for i in range(len(a) - 1):
#         if a[i] == a[i + 1]:
#             return False
#     return True
#
#
# def perm_alone(a: str):
#     perms = list(permutations(a))
#     perms = ["".join(elem for elem in tup) for tup in perms]
#     return len([x for x in perms if is_good(x) == True])
#
# a = "aa"
# print(perm_alone(a))


# 4
# def pairwise(vals, x):
#     res = 0
#
#     pairs = set()
#
#     for i in range(len(vals)):
#         for j in range(i, len(vals)):
#             if i != j and vals[i] + vals[j] == x and (vals[i], vals[j]) not in pairs:
#                 res += (i + j)
#                 pairs.add((vals[i], vals[j]))
#     return res
#
#
# print(pairwise([0, 0, 0, 0, 1, 1], 1))


# 5
# def bubble_sort(vals):
#     for i in range(len(vals) - 1):
#         for j in range(i, len(vals)):
#             if vals[i] > vals[j]:
#                 vals[i], vals[j] = vals[j], vals[i]
#     return vals
#
#
# vals = [5, 1, 3 , 8, 2, 6, 9, 4]

# print(bubble_sort(vals))


# 6

# def argmin(vals):
#     min = 1e9
#     idx = None
#     for i, val in enumerate(vals):
#         if vals[i] <= min:
#             min = vals[i]
#             idx = i
#     return idx


# def selection_sort(vals):
#     for i in range(len(vals)):
#         j = argmin(vals[i:])
#         vals[i], vals[i + j] = vals[i + j], vals[i]
#     return vals
#
#
# vals = [5, 1, 3, 8, 2, 6, 9, 4]
#
#
# print(selection_sort(vals))


# 7
# def insertion_sort(vals):
#     i = 0
#     j = 1
#     while j <= len(vals) - 1:
#         if vals[i] > vals[j]:
#             vals[i], vals[j] = vals[j], vals[i]
#             if i >= 1:
#                 i -= 1
#                 j -= 1
#                 continue
#         i += 1
#         j += 1
#
#     return vals
#
# vals = [5, 1, 3, 8, 2, 6, 9, 4]
#
# print(insertion_sort(vals))

# 8
# from copy import deepcopy

# vals = [5, 1, 3, 8, 2, 6, 9, 4]
#
#
# def merge(left, mid, right):
#     a = deepcopy(vals[left: mid])
#     b = deepcopy(vals[mid + 1: right])
#     i = 0
#     j = 0
#     c = []
#     while i < len(a) and j < len(b):
#         if a[i] < b[j]:
#             c.append(a[i])
#         else:
#             c.append(b[j])
#
#     while j < len(b):
#         c.append(b[j])
#
#     while i < len(a):
#         c.append(a[i])
#
#     vals[left:right] = c
#
#     return vals
#
#
# def merge_sort(left, right):
#     print(left, right)
#     if left < right:
#         mid = (left + right) // 2
#
#         merge_sort(left, mid)
#         merge_sort(mid + 1, right)
#
#         merge(left, mid, right)
#
#
# print(merge_sort(0, len(vals)))


# 9

# def quick_sort(vals):
#     return vals
#
# vals = [5, 1, 3 , 8, 2, 6, 9, 4]
#
# print(quick_sort(vals))


# 10
# def binary_search(vals):
#     return vals
#
# vals = [5, 1, 3 , 8, 2, 6, 9, 4]
#
# print(binary_search(vals))


# 11

# def get_final_open_doors(vals):
#     n = len(vals)
#     for i in range(n):
#         m = i + 1
#         for j in range(0, n, m):
#             print(m, j)
#             vals[j] = 1 - vals[j]
#     return vals
#
# vals = [0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0]
#
# print(get_final_open_doors(vals))


# 12
# from itertools import combinations, permutations
# import warnings
# warnings.filterwarnings("ignore")
#
# def equal_to(num: str, value: int = 24):
#     ops = ["+", "-", "/", "*", "(", ")", "(", ")", "(", ")"]
#     digits = [d for d in num]
#     all = ops + digits
#     final_res = []
#     for i in range(1, len(all)):
#         combs = combinations(all, i)
#         for comb in combs:
#             perms = permutations(comb)
#             for perm in perms:
#                 s = "".join(perm)
#                 res = None
#                 try:
#                     res = eval(s)
#                     # print(s)
#                 except:
#                     pass
#                 if isinstance(res, int):
#                     if res == value:
#                         final_res.append(res)
#                         print(len(final_res))
#                         # print(f"{s} evaluates to {value}")
#     return final_res


# print(equal_to("4878", 24))

# 13
import numpy as np

"""
6 = 11 ways
1 + 1 + 1 + 1 + 1 + 1 / 2 + 2 + 2 /

1 + 2 + 3 / 1 + 1 + 4 

/ 1 + 5 / 6 

3 + 3 / 2 + 4

1 + 1 + 1 + 1 + 2 / 1 + 1 + 1 + 3 / 

1 + 1 + 2 + 2

            1   - 1
          1   1   - 2
        1   1   1  - 3 
      1   2   1   1   -  4
    1   2   2   1   1   - 5  = 7
  1   3   3   2   1   1   - 6  = 11
1   3   3   2   2   1   1  - 7 =  13
"""


# def P(n: int):
#     mat = np.zeros((n, n))
#     mat[:, 0] = 1
#     for i in range(n):
#         mat[i, i] = 1
#         if i <= n - 2:
#             mat[i + 1, i] = 1
#     # mat[1:, 1] = 1
#     # mat[-1, :] = 1
#     # for i in range(2, n - 1):
#     #     for j in range(2, n - 1):
#     #         if i >= j:
#     #             mat[i, j] = j
#
#     print(mat)
#     print(sum(mat[-2, :]))
#
# print(P(6))
# 14
# from itertools import combinations, permutations

# # ABC problem
# def can_make_word(word, alphabet):
#     word = [w for w in word if "A" <= w <= "Z"]
#     if len(word) == 0:
#         return True
#     if len(word) % 2 == 1:
#         return False
#     groups = ["".join(word[i:i+2]) for i in range(0, len(word), 2)]
#     if len(groups) != len(set(groups)):
#         return False
#     oovocab_groups = [group for group in groups if group not in alphabet]
#     if len(oovocab_groups) == 0:
#         return True
#     return False
#     # used_chars = set()
#     # combs = list(combinations(word, len(word) // 2))
#     # print(combs)
#
#
# alphabet = ['BO', 'XK', 'DQ', 'CP', 'NA', 'GT', 'RE', 'TG', 'QD', 'FS', 'JW', 'HU', 'VI', 'AN', 'OB', 'ER', 'FS', 'LY',
#             'PC', 'ZM']
#
#
# word = "conFUSE"
# print(can_make_word(word, alphabet))

# 15
def sum_divisors(num: int):
    if num == 1 or num == 2:
        return 1
    s = 1
    for i in range(2, num // 2 + 1):
        if num % i == 0:
            s += i
    return s


#
#
# def abundant_deficient_perfect(num: int):
#     p = sum_divisors(num)
#     if p < num:
#         return 0
#     elif p == num:
#         return 1
#     elif p > num:
#         return 2
#
#
# def get_dpa(num: int):
#     res = [0, 0, 0]
#     for i in range(1, num + 1):
#         res[abundant_deficient_perfect(i)] += 1
#     return res
#
#
# print(get_dpa(10000))

# 16
# def acc(n):
#     def accumulator(x):
#         return x + n
#     return accumulator
#
# a = acc(3)
# b = a(-4)
#
# c = acc(b)
# d = c(1.5)
#
# e = acc(d)
# d = e(5)
#
# print(d)

# 17
# def ack(m, n):
#     if m == 0:
#         return n + 1
#     if m > 0 and n == 0:
#         return ack(m - 1, 1)
#     if m > 0 and n > 0:
#         return ack(m - 1, ack(m, n - 1))
#
# print(ack(1, 1))

# 18

# def format_text(text, just):
#
# val = [
#     'Given$a$text$file$of$many$lines',
#     'where$fields$within$a$line$',
#     'are$delineated$by$a$single$"dollar"$character',
#     'write$a$program',
#     'that$aligns$each$column$of$fields',
#     'by$ensuring$that$words$in$each$',
#     'column$are$separated$by$at$least$one$space.',
#     'Further,$allow$for$each$word$in$a$column$to$be$either$left$',
#     'justified,$right$justified',
#     'or$center$justified$within$its$column.'
# ]
#
# print(format_text(val, "left"))
# print(format_text(val, "center"))
# print(format_text(val, "right"))

# 19
#
# def amicable_pairs_up_to(n: int):
#     res = []
#     for i in range(1, n):
#         for j in range(i + 1, n + 1):
#             if sum_divisors(i) == j and sum_divisors(j) == i:
#                 res.append([i, j])
#     return res
#
# print(amicable_pairs_up_to(3000))

# 20
from collections import Counter

# def my_mode(vals):
#     cnt = Counter(vals)
#     maxi = max(cnt.values())
#     res = []
#     for (k, v) in cnt.items():
#         if v == maxi:
#             res.append(k)
#     return res
#
#
# vals = [1, 2, 3, 4, 2, 2, 3, 3, 5]
# print(my_mode(vals))

# 21
import math


# def pythagorean_means(values):
#     a = sum(values) / len(values)
#     prod = 1
#     for val in values:
#         prod *= val
#     g = math.pow(prod,  1 / len(values))
#
#     h = len(values) / sum([1 / val for val in values])
#
#     print("arithmetic", a)
#     print("geometric", g)
#     print("harmonic", h)
#     print("a >= g >= h", a >= g >= h)
# values = list(range(1, 11))
# print(pythagorean_means(values))

# 22
# def rms(values):
#     return math.sqrt(sum([val ** 2 for val in values]) / len(values))
#
#
# values = list(range(1, 11))
#
#
# print(rms(values))
# 23
# def endsin(a, b):
#     a = str(a)
#     b = str(b)
#     return a.endswith(b)
#
#
# def babbage(num):
#     for i in range(1, int(1e9)):
#         sqrd = i ** 2
#         if endsin(sqrd, num):
#             print(sqrd)
#             return i
#
# print(babbage(269_696))

# 24
# from queue import Queue
#
#
# def balanced_brackets(s):
#     q = []
#     for e in s:
#         if e == "[":
#             q.append(e)
#         if e == "]":
#             if len(q) == 0:
#                 return False
#             q.pop()
#
#     if len(q) == 0:
#         return True
#     return False
#
# s = "[[]][[][]]"
# print(balanced_brackets(s))

# 25
def euclid_dist(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


# def get_middle_point(p1, p2):
#     return [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]
#
#
# def get_circles(p1, p2, r):
#     if r == 0:
#         print("radius zero")
#
#     if p1 == p2:
#         print("the points are the same, thus the circles are the same")
#
#     dist = euclid_dist(p1, p2)
#
#     if dist > 2 * r:
#         print("circles are far away")
#     elif dist == 2 * r:
#         print("circles have one point of intersection")
#         from sympy import symbols, solve
#         x, y = symbols('x y')
#         # eq_1 = 2 * x + y
#         # eq_2 = x + 3 * y
#         eq_1 = x ** 2 - 2 * p1[0] * x + p1[0] ** 2 + y ** 2 - 2 * p1[1] * y + p1[1] ** 2 - r ** 2
#         eq_2 = x ** 2 - 2 * p2[0] * x + p2[0] ** 2 + y ** 2 - 2 * p2[1] * y + p2[1] ** 2 - r ** 2
#         print("eqs")
#         print(eq_1)
#         print(eq_2)
#         sol = solve((eq_1, eq_2), (x, y))
#
#         print(sol)
#     elif dist < 2 * r:
#         print("circles have two points")
#         pm = get_middle_point(p1, p2)
#         print(pm)
#
# import pandas as pd
# df = pd.read_csv("sample.csv")
# print(df)
#
# for i in range(len(df)):
#     p1 = [df.at[i, "p1x"], df.at[i, "p1y"]]
#     p2 = [df.at[i, "p2x"], df.at[i, "p2y"]]
#     r = df.at[i, "r"]
#     print(get_circles(p1, p2, r))

# 26
# from typing import List
#
# def brute_closest_points(points):
#     n = len(points)
#     if n < 2:
#         raise Exception("At least 2 points")
#     elif n == 2:
#         print(points)
#
#     min_dist = 1e9
#     min_points = []
#
#     for i in range(len(points) - 1):
#         for j in range(i + 1, len(points)):
#             dist = euclid_dist(points[i], points[j])
#             if dist < min_dist:
#                 min_dist = dist
#                 min_points = [points[i], points[j]]
#
#     print(min_dist, min_points)
#
#     return min_dist, min_points
#
#
# points = [[1, 2], [3, 3], [2, 2], [4, 4]]
#
# print(brute_closest_points(points))
#
# def eff_closest_pair(points):
#     n = len(points)
#     if n <= 3:
#         return brute_closest_points(points)
#     left_points = points[:n//2]
#     mid_points = points[n//2]
#     right_points = points[n//2:]
#
#     xp = [p[0] for p in points]
#     yp = [p[1] for p in points]
#
#     xl = [x[0] for x in left_points]
#     xm = mid_points[0]
#     xr = [x[0] for x in right_points]
#
#     yl = [x[1] for x in left_points]
#     ym = mid_points[1]
#     yr = [x[1] for x in right_points]
#
#     l_min, l_points = eff_closest_pair(left_points)
#     r_min, r_points = eff_closest_pair(right_points)
#
#     if l_min < r_min:
#         d_min = l_min
#         d_points = l_points
#     else:
#         d_min = r_min
#         d_points = r_points
#
#     ys = [points[i] for i in range(len(points)) if abs(mid_points[0] - points[i][0]) < d_min]
#     ns = len(ys)
#     closest = d_min
#     closest_pair = d_points
#     for i in range(1, ns):
#         k = i + 1
#         while k <= ns and ys[k][1] - ys[i][1] < d_min:
#             dist = euclid_dist(ys[k], ys[i])
#             if dist < closest:
#                 closest = dist
#                 closest_pair = [ys[k], ys[i]]
#             k += 1
#
#     return closest, closest_pair

# points = sorted(points, key=lambda x: x[1])
# print(eff_closest_pair(points))

# 27
# from itertools import combinations
#
#
# def combi(n, m):
#     return list(combinations(list(range(0, n)), m))
#
#
# print(combi(5, 3))


# 28
# def quibbling(input):
#     if len(input) == 0 and isinstance(input, list):
#         return "{}"
#     elif len(input) == 1:
#         return input[0]
#     elif len(input) == 2:
#         return input[0] + " and " + input[1]
#     else:
#         res = ""
#         for i in range(len(input) - 2):
#             res += (input[i] + ", ")
#         res += input[-2] + " and " + input[-1]
#         return res
#
# inputs = [[], ["ABC"], ["ABC", "DEF"], ["ABC", "DEF", "G", "H"]]
#
# for input in inputs:
#     print(quibbling(input))

# 29
# def az_sorted(values):
#     return sorted(values) == values
#
# def all_equal(values):
#     if len(values) == 0:
#         return True
#     return len(set(values)) == 1
#
# def compare_list_of_strings(values):
#     print(az_sorted(values), all_equal(values))
#
# tests = [['AA',  'BB', 'CC'], ['AA', 'ACB', 'AA'], [], ['AA']]
# for test in tests:
#     print(compare_list_of_strings(test))


# 30
# def convert_time(n):
#     min = 60
#     hour = 60 * min
#     day = hour * 24
#     week = 7 * day
#     steps = [week, day, hour, min, 1]
#     names = ["week", "day", "hour", "min", "sec"]
#     for (step, name) in zip(steps, names):
#         res = n // step
#         if res == 0:
#             continue
#         print(res, name)
#         n -= (res * step)
#
# print(convert_time(7259))
# print(convert_time(86400))
# print(convert_time(6000000))

# 31
# from copy import deepcopy
# def count_substring(big, small):
#     cnt = 0
#     i = 0
#     while i < len(big):
#         # print(i, cnt)
#         cur_cnt = deepcopy(cnt)
#         if big[i] == small[0]:
#             n = 1
#             for j in range(i + 1, min(i + len(small), len(big))):
#                 if big[j] == small[j - i]:
#                     n += 1
#             if n == len(small):
#                 cnt += 1
#         # print(i, cnt, cur_cnt)
#         i += 1
#         if cnt > cur_cnt:
#             i += len(small)
#             # print("increased i")
#     return cnt
#
#
# print(count_substring("abaabba*bbaba*bbab", "a*b"))

# 32
from itertools import combinations

# def count_coins(coins):
#     q = 25
#     d = 10
#     n = 5
#     p = 1
#     all = [q, d, n, p]
#     aux = []
#     for elem in all:
#         aux.append(coins // elem + 1)
#     final = [all[i] for i in range(len(all)) for _ in range(aux[i])]
#     # print(final)
#     res = 0
#     for i in range(1, max(aux)):
#         combs = list(set(list(combinations(final, i))))
#         for comb in combs:
#             # print(sum(comb))
#             if sum(comb) == coins:
#                 res += 1
#                 print(comb)
#     return res
#
# print(count_coins(15))

# 33
from copy import deepcopy


# to be done cramer todo
def get_delta(mat):
    """
    0,0  0,1, 0,2
    1,0  1,1  1,2
    2,0  2,1  2,2
    """
    if len(mat) == 3:
        return mat[0][0] * mat[1][1] * mat[2][2] + mat[0][1] * mat[1][2] * mat[2][0] + mat[1][0] * mat[2][1] * mat[0][2] - \
               mat[0][2] * mat[1][1] * mat[2][0] - mat[0][0] * mat[2][1] * mat[1][2] - mat[0][1] * mat[1][0] * mat[2][2]
    elif len(mat) == 2:
        return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]
    elif len(mat) == 1:
        return mat[0][0]
    else:
        raise Exception("Wrong mat len")


def cramer_rule(initial_mat, last_col):
    initial_mat = np.array(initial_mat)
    last_col = np.array(last_col)
    if len(initial_mat) >= 3 and len(initial_mat[0]) >= 3:
        mat = initial_mat[:3][:3]
        delta = get_delta(mat)
    elif len(initial_mat) < 3 or len(initial_mat[0]) < 3:
        mat = initial_mat[:2][:2]
        delta = get_delta(mat)

    if delta == 0:
        print("no crammer rule can be applied!")
        return

    print(delta)

    if len(initial_mat) == len(initial_mat[0]):
        print("squared matrix")
        res = []
        print(initial_mat)
        print(last_col)
        print("*" * 10)
        for i in range(len(last_col)):
            new_mat = deepcopy(initial_mat)
            new_mat[:, i] = last_col
            print(new_mat)
            res.append(get_delta(new_mat) / delta)
        return res
    else:
        print("rectangle matrix")


# print(cramer_rule([[3, 1, 1], [2, 2, 5], [1, -3, -4]], [3, -1, 2]))
# print(cramer_rule([[2, -1, 5, 1], [3, 2, 2, -6], [1, 3, 3, -1], [5, -2, -3, 3]], [-3, -32, -47, 49]))

# 34
# def stdev(vals):
#     mean_ = sum(vals) / (len(vals) )
#     return math.sqrt(sum([(x - mean_) ** 2 for x in vals]) / (len(vals)))
#
#
# vals = [2,4,4,4,5,5,7,9]
# print(stdev(vals))

# 35
# def cusip(input):
#     sum = 0
#     letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
#
#     for i, c in enumerate(input):
#         if c in "0123456789":
#             v = int(c)
#         elif c.upper() in letters:
#             p = letters.index(c.upper())
#             v = p + 9
#         elif c == "*":
#             v = 36
#         elif c == "@":
#             v = 37
#         elif c == "#":
#             v = 38
#
#         if (7 - i) % 2 == 0:
#             v = v * 2
#
#         sum = sum + int(v / 10) + v % 10
#
#     return (10 - (sum % 10)) % 10
#
# print(cusip("38259P50a"))

# 36
# import numpy
#
# from itertools import permutations
#
# def cut_rectangle(m, n):
#     if (m * n) % 2 == 1:
#         return -1
#
#     mat = numpy.zeros((m + 2, n + 2))
#
#     half = (m * n) // 2
#     to_split_values = [1] * half + [2] * half
#
#     def is_valid_permutation(perm, mat):
#         k = 0
#         one = dict()
#         two = dict()
#         labels = dict()
#         # populate matrix
#         for i in range(1, mat.shape[0] - 1):
#             for j in range(1, mat.shape[1] - 1):
#                 mat[i][j] = perm[k]
#                 if perm[k] == 1:
#                     one[k] = []
#                 if perm[k] == 2:
#                     two[k] = []
#                 labels[(i, j)] = k
#                 k += 1
#
#         for i in range(1, mat.shape[0] - 1):
#             for j in range(1, mat.shape[1] - 1):
#                 neighbors = [[-1, 0], [1, 0], [0, -1], [0, 1]]
#                 for neighbor in neighbors:
#                     neighbor_coords = [i + neighbor[0], j + neighbor[1]]
#                     for k in [1, 2]:
#                         # print(mat.shape, k, neighbor_coords)
#                         if mat[i, j] == k and mat[neighbor_coords[0], neighbor_coords[1]] == k:
#                             if k == 1:
#                                 one[labels[(i, j)]].append(labels[(neighbor_coords[0], neighbor_coords[1])])
#                             if k == 2:
#                                 two[labels[(i, j)]].append(labels[(neighbor_coords[0], neighbor_coords[1])])
#
#         def dfs(node, graph, viz):
#             if node not in viz:
#                 viz.add(node)
#                 neighbors = graph[node]
#                 for neighbor in neighbors:
#                     dfs(neighbor, graph, viz)
#
#         one_viz = set()
#         two_viz = set()
#
#         # print(len(two), len(one))
#         dfs(list(one.keys())[0], one, one_viz)
#         dfs(list(two.keys())[0], two, two_viz)
#
#         # print(len(one_viz), len(two_viz))
#         # print(mat.shape[0] * mat.shape[1])
#
#         if len(one) != len(one_viz) or len(two) != len(two_viz):
#             return False
#
#         if len(one_viz) != len(two_viz):
#             # print("falsch")
#             return False
#
#         if len(one_viz) + len(two_viz) != (mat.shape[0] - 2) * (mat.shape[1] - 2):
#             # print("falsch")
#             return False
#
#         # print("*" * 5)
#         # print(perm)
#         # print(labels)
#         # print(one, two, len(one), len(two), len(one_viz), len(two_viz))
#         """
#         0 1
#         2 3
#         """
#         return True
#
#     cnt = 0
#     perms = list(set(list(permutations(to_split_values))))
#
#     perms = perms[:len(perms) // 2]
#
#     for perm in perms:
#         if is_valid_permutation(perm, deepcopy(mat)):
#             cnt += 1
#     # print(len(perms))
#     return cnt
#
#
# print(cut_rectangle(2, 2))
# print(cut_rectangle(4, 3))
# print(cut_rectangle(4, 4))
# print(cut_rectangle(8, 3))
# print(cut_rectangle(7, 4))

# 37
# from datetime import datetime

# def get_date_format():
#     a = datetime.today().strftime('%Y-%m-%d')
#     parts = a.split("-")
#     parts = [str(int(part)) for part in parts]
#     a = "-".join(parts)
#     now = datetime.now()
#     day = now.strftime("%A")
#     month = now.strftime("%B")
#     day_num = parts[2]
#     year = parts[0]
#     b = f"{month}, {day} {day_num}, {year}"
#     return [a, b]
#
# print(get_date_format())

# 38
# from datetime import timedelta
#
# def add_12_hours(hour: str):
#     import dateutil.parser
#     yourdate = dateutil.parser.parse(hour)
#     print(type(yourdate))
#     print(yourdate)
#     return yourdate + timedelta(hours=12)
#
# print(add_12_hours("March 6 2009 7:30pm EST"))

# 39
# import datetime
#
#
# def day_of_the_week(first, last):
#     res = []
#     for year in range(first, last + 1):
#         d = datetime.date(year, 12, 25)
#         day = d.strftime("%A")
#         if day == "Sunday":
#             res.append(year)
#     return res
#
# print(day_of_the_week(2000, 2100))
# print(day_of_the_week(1970, 2017))
# print(day_of_the_week(2008, 2121))


# 40
# numbers = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
# symbols = ["C", "D", "H","S"]
# cards = []
# for num in numbers:
#     for sym in symbols:
#         cards.append(num + sym)
# print(cards)
#
# seed = 13
# def rng(seed):
#     new_seed = 214_013 * seed + 2_531_011 % 2 ^ 31
#     rand_ = seed / (2 ** 16)
#     return rand_
# print(rng(seed))

# 41
# deepcopy ---
# 42

# class Num:
#     def __init__(self, num):
#         self.num = num
#
# def primitive_data_type(num):
#     if isinstance(num, int) == False:
#         raise TypeError("Not a number")
#     if num < 1 or num > 10:
#         raise TypeError("Out of range")
#     return Num(num)
#
# print(primitive_data_type(4))
# print(primitive_data_type("test"))

# 43
# from itertools import combinations, permutations
#
# combs = list(combinations(list(range(1, 8)), 3))
#
# print(len(combs))
#
# cnt = 0
# for comb in combs:
#     # print(comb, sum(comb))
#     perms = list(permutations(comb))
#     for perm in perms:
#         if sum(perm) == 12 and perm[0] % 2 == 0:
#             print(perm)
#             cnt += 1
#
# print(cnt)
# 44
# TODO fix this shit
def discordian_date(date_obj: str):
    gregorian_year = date_obj.year
    gregorian_month = date_obj.month
    gregorian_day = date_obj.day

    total_days = gregorian_day + gregorian_month * 30 + gregorian_year * 365
    year = None
    print(f"{weekday}, the {day_num}th day of {season} in the YOLD {year}")


import datetime

seasons = ['Chaos', 'Discord', 'Confusion', 'Bureaucracy', 'The Aftermath']
weekdays = ['Sweetmorn', 'Boomtime', 'Pungenday', 'Prickle-Prickle', 'Setting Orange']
apostles = ['Mungday', 'Mojoday', 'Syaday', 'Zaraday', 'Maladay']
holidays = ['Chaoflux', 'Discoflux', 'Confuflux', 'Bureflux', 'Afflux']

datetime_obj = datetime.date(2010, 6, 22)


# print(discordian_date(datetime_obj))

# 45
# def dot_product(a, b):
#     return sum([i * j for (i, j) in zip(a, b)])
#
#
# print(dot_product([1, 3, -5], [4, -2, -1]))

# 46
#
# options = ["addition", "subtraction", "multiplication", "division", "exponentiation"]
#
# def operation(option, a, b):
#     sm, op = option.split("_")
#     if op == "add":
#         if sm == "s":
#             for i in range(len(a)):
#                 for j in range(len(a[i])):
#                     a[i][j] += b
#         elif sm == "m":
#             for i in range(len(a)):
#                 for j in range(len(a[i])):
#                     a[i][j] += b[i][j]
#     if op == "sub":
#         if sm == "s":
#             for i in range(len(a)):
#                 for j in range(len(a[i])):
#                     a[i][j] -= b
#         elif sm == "m":
#             for i in range(len(a)):
#                 for j in range(len(a[i])):
#                     a[i][j] -= b[i][j]
#     if op == "mult":
#         if sm == "s":
#             for i in range(len(a)):
#                 for j in range(len(a[i])):
#                     a[i][j] *= b
#         elif sm == "m":
#             for i in range(len(a)):
#                 for j in range(len(a[i])):
#                     a[i][j] *= b[i][j]
#     if op == "div":
#         if sm == "s":
#             for i in range(len(a)):
#                 for j in range(len(a[i])):
#                     a[i][j] /= b
#         elif sm == "m":
#             for i in range(len(a)):
#                 for j in range(len(a[i])):
#                     a[i][j] /= b[i][j]
#     if op == "exp":
#         if sm == "s":
#             for i in range(len(a)):
#                 for j in range(len(a[i])):
#                     a[i][j] = math.pow(a[i][j], b)
#         elif sm == "m":
#             for i in range(len(a)):
#                 for j in range(len(a[i])):
#                     a[i][j] = math.pow(a[i][j], b[i][j])
#     return a
#
# print(operation("m_add",[[1,2],[3,4]],[[1,2],[3,4]]))
# print(operation("s_add",[[1,2],[3,4]],2))
# print(operation("m_sub",[[1,2],[3,4]],[[1,2],[3,4]]))
# print(operation("m_mult",[[1,2],[3,4]],[[1,2],[3,4]]))
# print(operation("m_div",[[1,2],[3,4]],[[1,2],[3,4]]))
# print(operation("m_exp",[[1,2],[3,4]],[[1,2],[3,4]]))
# print(operation("m_add",[[1,2,3,4],[5,6,7,8]],[[9,10,11,12],[13,14,15,16]]))

# 47
def is_prime(n):
    for i in range(2, n // 2):
        if n % i == 0:
            return False
    return True


#
#
# def is_emirp(n):
#     return is_prime(n) and is_prime(int(str(n)[::-1]))
#
#
# def emirps(n=None, boundaries=None):
#     if n is not None:
#         cnt = 0
#         for i in range(2, int(1e9)):
#             if is_emirp(i):
#                 print(i)
#                 cnt += 1
#                 if cnt == n:
#                     break
#     if boundaries is not None:
#         for i in range(boundaries[0], boundaries[1] + 1):
#             if is_emirp(i):
#                 print(i)
#
# print(emirps(n=20))
# print(emirps(boundaries=[7700, 8000]))


# 48

from collections import Counter
import math


def entropy(n):
    c = Counter(n)
    h = 0
    for (k, v) in c.items():
        cur = v / len(n) * math.log(v / len(n), 2)
        h += cur
    return round(h * (-1), 8)


#
#
# print(entropy("1223334444"))
#

# 49
# def equilibrium_index(a):
#     for i in range(1, len(a) - 1):
#         l = sum(a[:i])
#         r = sum(a[i + 1:])
#         if l == r:
#             print(i)
#
#     if a[0] == sum(a[1:]):
#         return 0
#
#     if a[-1] == sum(a[:-1]):
#         return len(a) - 1
#
#
# x = [-7, 1, 5, 2, -4, 3, 0]

# x = [0, 1, 2, 3, -6]

# print(equilibrium_index(x))

# 50
# def eth_mult(a, b):
#     halves = []
#     doubles = []
#
#     while a != 0:
#         halves.append(a)
#         a //= 2
#         doubles.append(b)
#         b *= 2
#
#     to_be_removed_doubles = []
#     to_be_removed_halves = []
#
#     for (half, double) in zip(halves, doubles):
#         if half % 2 == 0:
#             to_be_removed_doubles.append(double)
#             to_be_removed_halves.append(half)
#
#     for (double, half) in zip(to_be_removed_doubles, to_be_removed_halves):
#         doubles.remove(double)
#         halves.remove(half)
#
#     return sum(doubles)
#
# a = 17
# b = 34
# print(eth_mult(a, b))


# 51
# def eulers_method(t0, tr, k, t):
#     res = tr + (t0 - tr) * math.pow(math.e, (-k * t))
#     print((t0 - tr) * math.pow(math.e, (-k * t)))
#     return res
#
# print(eulers_method(0, 100, 100, 2))
# print(eulers_method(0, 100, 100, 5))
# print(eulers_method(0, 100, 100, 10))

# 52
# from itertools import combinations
#
# def binom(n, k):
#     return len(list(combinations(list(range(n)), k)))
#
#
# print(binom(10, 4))

# 53
# def conv_arrow_rule(rule):
#     first_arrow = rule.find("->")
#     elems = [rule[:first_arrow].strip(), rule[first_arrow + 2:].strip()]
#     return elems
#
#
# def apply_rule(data, rule):
#     elements = conv_arrow_rule(rule)
#     first, second = elements
#     return data.replace(first, second)
#
#
# from copy import deepcopy
#
#
# def markov(rules_list, data_list, outputs):
#     for (rules, data, output) in zip(rules_list, data_list, outputs):
#         initial_output = deepcopy(data)
#         while True:
#             are_there_any_rules_that_can_be_applied = False
#             for rule in rules:
#                 elements = conv_arrow_rule(rule)
#                 pattern, replacement = elements
#                 if pattern in data:
#                     are_there_any_rules_that_can_be_applied = True
#
#             if are_there_any_rules_that_can_be_applied is False:
#                 break
#
#             first_rule = None
#             min_idx = len(data) * 999
#             for rule in rules:
#                 elements = conv_arrow_rule(rule)
#                 pattern, replacement = elements
#                 # print(pattern, len(pattern), replacement, len(replacement))
#                 find_idx = data.find(pattern)
#                 if find_idx != -1:
#                     if find_idx < min_idx:
#                         find_idx = min_idx
#                         first_rule = rule
#             if first_rule is None:
#                 continue
#             # print(min_idx, first_rule, len(rules), rules)
#             rules.remove(first_rule)
#             # print(min_idx, first_rule, len(rules), rules)
#             data = apply_rule(data, first_rule)
#             print(first_rule, data)
#         print("*" * 10)
#         print("initial: ", initial_output)
#         print("after markov: ", data)
#         print("gold standard: ", output)
#         print(data == output)
#         print("*" * 10)
#
#
# def main():
#     rules = [
#         [
#             "A -> apple", "B -> bag", "S -> shop", "T -> the",
#             "the shop -> my brother", "a never used -> .terminating rule"
#         ], [
#             "A -> apple", "B -> bag", "S -> shop", "T -> the",
#             "the shop -> my brother", "a never used -> .terminating rule"
#         ], [
#             "A -> apple", "WWWW -> with", "Bgage -> ->.*", "B -> bag",
#             "->.* -> money", "W -> WW", "S -> .shop", "T -> the",
#             "the shop -> my brother", "a never used -> .terminating rule"
#         ], [
#             "_+1 -> _1+", "1+1 -> 11+", "1! -> !1", ",! -> !+", "_! -> _", "1*1 -> x,@y", "1x -> xX",
#             "X, -> 1,1", "X1 -> 1X", "_x -> _X", ",x -> ,X", "y1 -> 1y", "y_ -> _", "1@1 -> x,@y",
#             "1@_ -> @_", ",@_ -> !_", "++ -> +", "_1 -> 1", "1+_ -> 1", "_+_ -> "
#         ], [
#             "A0 -> 1B", "0A1 -> C01", "1A1 -> C11", "0B0 -> A01", "1B0 -> A11",
#             "B1 -> 1B", "0C0 -> B01", "1C0 -> B11", "0C1 -> H01", "1C1 -> H11"
#         ]]
#
#     data = ["I bought a B of As from T S.",
#             "I bought a B of As from T S.",
#             "I bought a B of As W my Bgage from T S.",
#             "_1111*11111_",
#             "000000A000000"
#             ]
#
#     outputs = [
#         "I bought a bag of apples from my brother.",
#         "I bought a bag of apples from T shop.",
#         "I bought a bag of apples with my money from T shop.",
#         "11111111111111111111",
#         "00011H1111000"
#     ]
#     print(markov(rules, data, outputs))
#
#
# if __name__ == "__main__":
#     main()

# 54


# def look_for_closing_bracket(input, index):
#     for i in range(index + 1, len(input) - 1):
#         if input[i] == "]":
#             return i
#
# def look_for_open_bracket(input, index):
#     for i in range(index - 1, 0, -1):
#         if input[i] == "[":
#             return i


# def execute_brainfuck(input):
#     pointer = 0
#     memory_cell = [0] * len(input)
#     for i, c in enumerate(input):
#         if c == ">":
#             pointer += 1
#         elif c == "<":
#             pointer -= 1
#         elif c == "+":
#             memory_cell[pointer] += 1
#         elif c == "-":
#             memory_cell[pointer] -= 1
#         elif c == ".":
#             print(memory_cell[pointer])
#         elif c == ",":
#             new_character = input("Please enter a character:")
#             input[pointer] = new_character[0]
#         elif c == "[":
#             if memory_cell[pointer] == 0:
#                 idx = look_for_closing_bracket(input, i)
#                 pointer = idx + 1
#         elif c == "]":
#             if memory_cell[pointer] != 0:
#                 idx = look_for_open_bracket(input, i)
#                 pointer = idx - 1
#
#
# print(execute_brainfuck("++++++[>++++++++++<-]>+++++."))

# 55

# def prime_generator(n=None, boundaries=None):
#     if n is not None:
#         cnt = 0
#         for i in range(2, int(1e9)):
#             if is_prime(i):
#                 print(i)
#                 cnt += 1
#                 if cnt == n:
#                     break
#     if boundaries is not None:
#         for i in range(boundaries[0], boundaries[1] + 1):
#             if is_prime(i):
#                 print(i)
#
# print(prime_generator(n=20))
# print(prime_generator(boundaries=[7700, 8000]))

# 56
# def fibonacci_word(n):
#     res = []
#     for i in range(1, n + 1):
#         elem = dict()
#         elem["N"] = i
#         if i == 1:
#             elem["Word"] = "1"
#         elif i == 2:
#             elem["Word"] = "0"
#         else:
#             elem["Word"] = res[i - 2]["Word"] + res[i - 3]["Word"]
#
#         elem["Length"] = len(elem["Word"])
#         elem["Entropy"] = entropy(elem["Word"])
#         res.append(elem)
#
#     return res
#
#
# print(fibonacci_word(5))
# print(fibonacci_word(7))

# 57
# def fizz_buzz():
#     res = []
#     for i in range(1, 101):
#         if i % 3 == 0 and i % 5 == 0:
#             res.append("FizzBuzz")
#         elif i % 5 == 0:
#             res.append("Buzz")
#         elif i % 3 == 0:
#             res.append("Fizz")
#     return res
#
# print(fizz_buzz())

# 58
# TODO figure this shit out
# def fractran(input):
#     fractions = [[int(e) for e in fraction.split("/")] for fraction in input.split(",")]
#     print(fractions)
#
#
# print(fractran("3/2, 1/3"))
# print(fractran("3/2, 5/3, 1/5"))
# print(fractran("3/2, 6/3"))
# print(fractran("2/7, 7/2"))
# print(fractran("17/91, 78/85, 19/51, 23/38, 29/33, 77/29, 95/23, 77/19, 1/17, 11/13, 13/11, 15/14, 15/2, 55/1"))


# 59
def factorial(n):
    if n == 0:
        return 1
    if n == 1:
        return 1
    # print(n)
    return n * factorial(n - 1)


#
# print(factorial(5))
# 60
def convert_to_binary(p):
    if p >= 0:
        return bin(p)[2:]
    else:
        return bin(p)[3:]


# def mod_pow(p, diver):
#     binary = convert_to_binary(p)
#     mod = 1
#     for i, top in enumerate(binary):
#         bit = binary[i + 1:]
#         squared = mod * mod
#         if top == "1":
#             multiplied = 2 * squared
#         elif top == "0":
#             multiplied = None
#         else:
#             raise Exception("Wrong")
#
#         if multiplied is None:
#             mod = squared % diver
#         else:
#             mod = multiplied % diver
#
#         # print(bit, top, mod, multiplied, squared)
#
#     if mod == 1:
#         return True
#     return False
#
# def check_mersenne(p):
#     n = 2 ** p - 1
#     square_root = math.sqrt(n)
#     k = 0
#     potential_divider = 2 * k * p + 1
#     while potential_divider <= square_root:
#         is_divided = mod_pow(p, potential_divider)
#         if is_divided:
#             return f"2 ^ {p} - 1 divides with {potential_divider}"
#         k += 1
#         potential_divider = 2 * k * p + 1
#
#     return f"2 ^ {p} - 1 is prime"
#
# print(check_mersenne(3))
# print(check_mersenne(23))
# print(check_mersenne(929))


# 61
# def get_factors(n):
#     res = [1, n]
#     for i in range(2, n // 2):
#         res.append(i)
#     return res
#
# print(get_factors(45))

# 62
def nth_fibo(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return nth_fibo(n - 1) + nth_fibo(n - 2)


#
# print(nth_fibo(10))

# 63
# def nth_lucas_fibo(n, k, first = 1, second = 2):
#     if n == 2:
#         if k == 0:
#             return first
#         elif k == 1:
#             return second
#         else:
#             return nth_lucas_fibo(n, k - 1) + nth_lucas_fibo(n, k - 2)
#     elif n == 3:
#         if k == 0:
#             return first
#         elif k == 1:
#             return second
#         elif k == 2:
#             return first + second
#         else:
#             return nth_lucas_fibo(n, k - 1) + nth_lucas_fibo(n, k - 2) + nth_lucas_fibo(n, k - 3)
#     elif n == 4:
#         if k == 0:
#             return first
#         elif k == 1:
#             return second
#         elif k == 2:
#             return first + second
#         elif k == 3:
#             return first + second  + second
#         else:
#             return nth_lucas_fibo(n, k - 1) + nth_lucas_fibo(n, k - 2) + nth_lucas_fibo(n, k - 3) + nth_lucas_fibo(n, k - 4)
#     else:
#         if k < n:
#             return nth_lucas_fibo(n - 1, k)
#         else:
#             total = 0
#             for i in range(1, n + 1):
#                 total += nth_lucas_fibo(n, k - i)
#             return total
#
#
# def fib_luc(n, k, seq_type):
#     if seq_type == "f":
#         return nth_lucas_fibo(n, k, 1, 1)
#     elif seq_type == "l":
#         return nth_lucas_fibo(n, k, 2, 1)
#
#
# print(fib_luc(2, 10, "f"))
# print(fib_luc(3, 15, "f"))
# print(fib_luc(4, 15, "f"))
# print(fib_luc(2, 10, "l"))
# print(fib_luc(3, 15, "l"))
# print(fib_luc(4, 15, "l"))
# print(fib_luc(5, 15, "l"))

# 64
# def reduce_ratio(a, b):
#     pass
#
#
# def farey_sequence(n: int):
#     if n == 1:
#         return []
#     else:
#         return sorted([f"{i}/{n}" for i in range(1, n)] + farey_sequence(n - 1), key=lambda x: eval(x))
#
# print(farey_sequence(3))
# print(farey_sequence(4))
# print(farey_sequence(5))

# 65
# def maximum_subseq(vals):
#     sums = []
#     s = 0
#     neg_cnt = 0
#     for val in vals:
#         s += val
#         if val < 0:
#             neg_cnt += 1
#         sums.append(s)
#
#     if neg_cnt == len(vals):
#         return []
#
#     # print(sums)
#
#     maximal = 0
#
#     cur_l = None
#     cur_r = None
#
#     max_l = None
#     max_r = None
#     # print(vals)
#     # print(sums)
#     for i in range(len(sums)):
#         if sums[i] >= 0:
#             if cur_l is None:
#                 cur_l = i
#             cur_r = i
#             if (cur_l >= 1 and sums[cur_r] - sums[cur_l - 1] > maximal) or (cur_l == 0 and sums[cur_r] > maximal):
#                 if cur_l >= 1:
#                     maximal = sums[cur_r] - sums[cur_l - 1]
#                 else:
#                     maximal = sums[cur_r]
#                 max_l = cur_l
#                 max_r = cur_r
#         else:
#             cur_l, cur_r = None, None
#
#     return vals[max_l: max_r + 1]
#
# print(maximum_subseq([1, 2, -1, 3, 10, -10]))
# print(maximum_subseq([0, 8, 10, -2, -4, -1, -5, -3]))
# print(maximum_subseq([9, 9, -10, 1]))
# print(maximum_subseq([7, 1, -5, -3, -8, 1]))
# print(maximum_subseq([-3, 6, -1, 4, -4, -6]))
# print(maximum_subseq([-1, -2, 3, 5, 6, -2, -1, 4, -4, 2, -1]))

# 66
def gcd(a, b):
    # print(a, b/)
    if a == b:
        return a
    while a != 1 and b != 1:
        # print(a, b)
        if a > b:
            c = a - b
        else:
            c = b - a
        if a % c == 0 and b % c == 0:
            return c
        if a > b:
            a = c
        else:
            b = c


#
#
# print(gcd(24, 36))
# print(gcd(30, 48))


# 67
# def gray_code(b):

# 68

# def is_square(num):
#     rad = math.sqrt(num)
#     return rad == int(rad)
#
import math


def is_cube(num):
    if num == 1:
        return True
    for i in range(1, int(math.sqrt(num))):
        if i * i * i == num:
            return True
    return False


# def generator_exponential(n):
#     k = 0
#     num = 1
#     res = []
#     while k < n:
#         if is_square(num) == True and is_cube(num) == False:
#             res.append(num)
#             k += 1
#         num += 1
#     return res
#
# print(generator_exponential(7))
# print(generator_exponential(10))

# 69
# import string
# def generate_ascii(l, r):
#     letters = string.ascii_lowercase
#     res = []
#     for let in letters:
#         if l <= let <= r:
#             res.append(let)
#     return res
#
# print(generate_ascii("c", "g"))

# 70
# def general_fizz_buzz(fizz_buzz_mat, num):
#     n1 = fizz_buzz_mat[0][0]
#     s1 = fizz_buzz_mat[0][1]
#
#     n2 = fizz_buzz_mat[1][0]
#     s2 = fizz_buzz_mat[1][1]
#
#     if num % n1 == 0 and num % n2 == 0:
#         return s1 + s2
#     elif num % n1 == 0:
#         return s1
#     elif num % n2 == 0:
#         return s2
#     else:
#         return ""
#
# print(general_fizz_buzz([[3, "Fizz"], [5, "Buzz"]], 6))

# 71
# TODO
# def gaussian_elimination(A, b):
#     pass
# print(gaussian_elimination([[1,1], [1,-1]], [5,1]))

# 72
# from scipy.integrate import quad
#
# def gamma_f_aux(x, t):
#     return t ** (x - 1) * math.e ** (-x)
#
#
# def gamma_f(x):
#     return quad(gamma_f_aux, 0, np.inf, args=x)[0]
#
#
# print(gamma_f(.1))
# print(gamma_f(.2))
# print(gamma_f(.3))
# print(gamma_f(.4))
# print(gamma_f(.5))

# 73
def hailstone(n):
    if n == 1:
        return 1
    elif n % 2 == 0:
        # print(n)
        return 1 + hailstone(n // 2)
    elif n % 2 == 1:
        # print(n)
        return 1 + hailstone(3 * n + 1)


def hailstone_aux(n):
    mini = n
    maxi = n
    level = hailstone(n)

    while mini != 0:
        if hailstone(mini) == level:
            mini -= 1
        else:
            break

    while maxi != int(1e9):
        if hailstone(maxi) == level:
            maxi += 1
        else:
            break

    return [mini, maxi]


# TODO figure it out
# print(hailstone_aux(30))
# 74

# def hash_from_arrays(a, b):
#     dic = dict()
#     for (i, j) in zip(a, b):
#         dic[i] = j
#     return dic
#
# print(hash_from_arrays([1, 2, 3, 4, 5], ["a", "b", "c", "d", "e"]))

# 75

def get_digits(n):
    return list(str(n))


#
# def is_harshad(n):
#     digits = [int(digit) for digit in get_digits(n)]
#     return n % sum(digits) == 0
#
#
# def generater_harshad(n):
#     k = 0
#     num = n + 1
#     res = []
#     while k < 10:
#         if is_harshad(num):
#             k += 1
#             res.append(num)
#         num += 1
#     return res
#
#
# print(generater_harshad(400))

# 76
def num_digits(n):
    return len(str(n))


def get_digits(n):
    return [int(d) for d in str(n)]


def sum_digits(n):
    return sum(get_digits(n))


# def happy_number(n):
#     while num_digits(n) != 1:
#         n = sum_digits(n)
#     if n == 1:
#         return True
#     return False
#
# print(happy_number(1))
# print(happy_number(2))
# print(happy_number(7))
# print(happy_number(10))
# print(happy_number(13))
# print(happy_number(19))

# 77


# TODO
# def hofstadter(n):
#     aux = [-1] * (n + 4)
#     aux[0] = 1
#     aux[1] = 1
#     for i in range(2, n + 2):
#         aux[i] = aux[i - aux[i - 1]] + aux[i - aux[i - 2]]
#     return aux[n - 1]
#
# print(hofstadter(2500))


# 78

# ffs_terms = [0] * int(1e6)
# ffr_terms = [0] * int(1e6)
#
# ffr_terms[1] = 1
# ffs_terms[1] = 2

# def ffr(n):
#     global ffr_terms, ffs_terms
#     if n == 1:
#         return ffr_terms[1]
#     else:
#         if ffr_terms[n - 1] == 0 and ffs_terms[n - 1] == 0:
#             return ffr(n - 1) + ffs(n - 1)
#         return ffr_terms[n - 1] + ffs_terms[n - 1]
#
#
# def get_first_r_terms(n):
#     global ffr_terms, ffs_terms
#     res = []
#     for i in range(1, n + 1):
#         if ffr_terms[i] == 0:
#             value = ffr(i)
#             ffr_terms[value] = value
#         res.append(ffr_terms[i])
#     return res
#
# def ffs(n):
#     global ffr_terms, ffs_terms
#     if n == 1:
#         return ffs_terms[1]
#     else:
#         if ffs_terms[n] == 0:
#             r_terms = get_first_r_terms(n)
#             k = 1
#             num = 3
#             while k < n:
#                 if num not in r_terms:
#                     k += 1
#                 num += 1
#             ffs_terms[n] = num
#         return ffs_terms[n]
#
# print(ffr(11))
# print(ffr(50))
# print(ffr(100))
# print(ffr(1000))
# print(ffs(10))
# print(ffs(50))
# print(ffs(100))
# print(ffs(1000))

# 79
from tqdm import tqdm
#
# def heron(a, b, c):
#     s = (a + b + c) / 2
#     # print((s * (s - a) * (s - b) * (s - c)))
#     return math.sqrt((s * (s - a) * (s - b) * (s - c)))
#
#
# def is_heronian(a, b, c):
#     area = heron(a, b, c)
#     return int(area) == area
#
#
# def can_be_triangle(a, b, c):
#     return a + b > c and b + c > a and a + c > b
#
# triplets = set()
#
# def get_herons_triangles(n):
#     cnt = 0
#     res = []
#     m = n * 2
#     while True:
#         for i in range(1, m):
#             for j in range(1, m):
#                 for k in range(1, m):
#                     if can_be_triangle(i, j, k):
#                         if gcd(i, j) == 1 and gcd(j, i) == 1 and gcd(j, k) == 1:
#                             if is_heronian(i, j, k):
#                                 if tuple(sorted((i, j, k))) not in triplets:
#                                     triplets.add(tuple(sorted((i, j, k))))
#                                     cnt += 1
#                                     res.append([i, j, k])
#                                     print(cnt)
#                                     if cnt >= n:
#                                         return res
#         break
#
#
# print(get_herons_triangles(25))

# 80
# def hash():
#     pass
#
#
# def join():
#     pass
#
#
# def hash_join(a, b):
#     a_id_col = "name"
#     b_id_col = "character"
#     c = []
#     for i, row in enumerate(b):
#         elem = dict()
#         for k, v in row.items():
#             elem[f"b_{k}"] = v
#         c.append(elem)
#
#     for i, row_a in enumerate(a):
#         a_id = row_a[a_id_col]
#         rows_c, js_c = [], []
#         for j, row_c in enumerate(c):
#             c_id = row_c[f"b_{b_id_col}"]
#
#             if a_id == c_id:
#                 rows_c.append(row_c)
#                 js_c.append(j)
#
#         for match, (j, row_c) in enumerate(zip(js_c, rows_c)):
#             if match == 0:
#                 for k, v in row_a.items():
#                     c[j][f"a_{k}"] = v
#             else:
#                 for k, v in row_a.items():
#                     row_c[f"a_{k}"] = v
#                 c.append(row_c)
#
#     for i, row_c in enumerate(c):
#         print(row_c)
#         print(i)
#         if i == 50:
#             break
#         c_id = row_c[f"b_{b_id_col}"]
#         rows_a, js_a = [], []
#         for j, row_a in enumerate(a):
#             a_id = row_a[a_id_col]
#
#             if a_id == c_id:
#                 rows_a.append(row_a)
#                 js_a.append(j)
#
#         for match, (j, row_a) in enumerate(zip(js_a, rows_a)):
#             if match == 0:
#                 to_add_k_v = []
#                 for k, v in row_c.items():
#                     to_add_k_v.append((f"{k}", v))
#                 for k, v in to_add_k_v:
#                     c[j][f"{k}"] = v
#             else:
#                 for k, v in row_c.items():
#                     row_a[f"{k}"] = v
#                 c.append(row_a)
#
#     c = [str(elem) for elem in c]
#     c = list(set(c))
#
#     print(len(c))
#     for elem in c:
#         print(elem)
#     # return c
#
# print(hash_join([{ "age": 27, "name": "Jonah" }, { "age": 18, "name": "Alan" }, { "age": 28, "name": "Glory" },
#                   { "age": 18, "name": "Popeye" }, { "age": 28, "name": "Alan" }],
#                 [{ "character": "Jonah", "nemesis": "Whales" }, { "character": "Jonah", "nemesis": "Spiders" },
#                  { "character": "Alan", "nemesis": "Ghosts" }, { "character":"Alan", "nemesis": "Zombies" },
#                  { "character": "Glory", "nemesis": "Buffy" }]))

# 81
import string

# def is_valid(iban):
#     country_code = iban[:2].strip()
#     for c in country_code:
#         if c not in string.ascii_uppercase:
#             return False
#
#     check_digits = iban[2:4]
#
#     bban = iban[4:].strip().replace(" ", "")
#
#     letters = [str(elem) for elem in list(range(10))] + list(string.ascii_uppercase)
#
#     country_code = "".join([str(letters.index(c)) for c in country_code])
#
#     bank_name = bban[:4]
#     bban = bban[4:]
#
#     # print(bank_name, len(bank_name))
#     bank_name = "".join([str(letters.index(c)) for c in bank_name])
#
#     # print(bank_name)
#     # print(country_code)
#     # print(check_digits)
#     # print(bban)
#
#     iban = bank_name + bban + country_code + check_digits
#
#     try:
#         iban = int(iban)
#     except ValueError:
#         iban = float(iban)
#
#     if int(iban) % 97 == 1:
#         return True
#     return False
#
# print(is_valid("GB82 WEST 1234 5698 7654 32"))
# print(is_valid("GB82 WEST 1.34 5698 7654 32"))
# print(is_valid("GB82 WEST 1234 5698 7654 325"))
# print(is_valid("GB82 TEST 1234 5698 7654 32"))
# print(is_valid("SA03 8000 0000 6080 1016 7519"))


# 82
# def im(n):
#     mat = [[0 for i in range(n)] for j in range(n)]
#     for i in range(len(mat)):
#         mat[i][i] = 1
#     return mat
#
# print(im(3))

# 83


# def iterated_square(n):
#     digits = get_digits(n)
#     res = 0
#     for d in digits:
#         res += d ** 2
#
#     if res == 1:
#         return 1
#     elif res == 89:
#         return 89
#     else:
#         return iterated_square(res)
#
#
# print(iterated_square(20))
# print(iterated_square(70))

# 84

# def IEC(word):
#
# def EIC(word):

# def IBeforeExceptC(word):
#     if "cei" in word:
#         return True
#     if "cie" in word:
#         return False
#
#     # IEC(word)
#     # EIC(word)
#
#
# print(IBeforeExceptC("receive"))
# print(IBeforeExceptC("science"))
# print(IBeforeExceptC("imperceivable"))
# print(IBeforeExceptC("inconceivable"))
# print(IBeforeExceptC("insufficient"))
# print(IBeforeExceptC("omniscient"))

# 85
# def josephus(n, k):
#     prisoners = list(range(n))
#     print(prisoners)
#     num_killed = 0
#     while num_killed <= n - 1:
#         to_remove = []
#         if len(prisoners) > k:
#             for i in range(0, len(prisoners), k):
#                 to_remove.append(prisoners[i])
#         else:
#             to_remove = prisoners[:-1]
#         # print(to_remove)
#         for prisoner in to_remove:
#             prisoners.remove(prisoner)
#             num_killed += 1
#             if num_killed == n - 1:
#                 return prisoners[0]
#
# print(josephus(5, 2))
# print(josephus(30, 3))
# print(josephus(30, 5))
# print(josephus(20, 2))
# print(josephus(17, 6))
# print(josephus(29, 4))

# 86
# def jortsort(arr):
#     return sorted(arr) == arr
#
# print(jortsort([1,2,3,4,5]))
# print(jortsort([1,2,13,4,5]))
# print(jortsort([12,4,51,2,4]))
# print(jortsort([1,2]))
# print(jortsort([5,4,3,2,1]))
# print(jortsort([1,1,1,1,1]))

# 87
# def num_matching_characters(s1, s2):
#     max_dist = int(max(len(s1), len(s2)) / 2) - 1
#     m = 0
#     for i in range(len(s1)):
#         for j in range(len(s2)):
#             if s1[i] == s2[j] and abs(i - j) <= max_dist:
#                 m += 1
#     return m
#
# def num_transpositions(s1, s2, m):
#     max_dist = int(max(len(s1), len(s2)) / 2) - 1
#     t = 0
#     for i in range(len(s1)):
#         for j in range(len(s2)):
#             for p in range(len(s1)):
#                 for q in range(len(s2)):
#                     if s1[i] == s2[j] and abs(i - j) <= max_dist and s1[p] == s2[q] and abs(p - q) <= max_dist \
#                         and i != p and j != q:
#                         t += 1
#     return t
#
#
# def jaro(s1, s2):
#     m = num_matching_characters(s1, s2)
#     t = num_transpositions(s1, s2, m)
#
#     if m == 0:
#         return 0
#     else:
#         return (1/3) * ( m / len(s1) + m / len(s2) + (m - t) / m)
#
# print(jaro("DWAYME", "DUANE"))
# print(jaro("MARTHA", "MARHTA"))
# print(jaro("DIXON", "DICKSONX"))
# print(jaro("JELLYFISH", "SMELLYFISH"))
# print(jaro("HELLOS", "CHELLO"))
# print(jaro("ABCD", "BCDA"))


# 88
# def convert_to_base(n, b):
#     if n == 0:
#         return [0]
#     digits = []
#     while n:
#         digits.append(int(n % b))
#         n //= b
#     # print(digits[::-1])
#     num = 0
#     for digit in digits[::-1]:
#         num = num * 10 + digit
#
#     return num  # digits[::-1]
#
# def isKaprekar(n, bs):
#     if n <= 9:
#         return True
#     n = convert_to_base(n, bs)
#     # print(n)
#     sq = n ** 2
#     digits = str(sq)
#     for i in range(1, len(digits)):
#         a = int(digits[:i])
#         b = int(digits[i:])
#         # print(a, b, a + b, sq)
#         if a + b == n:
#             return True
#     return False
#
#
# print(isKaprekar(1, 10))
# print(isKaprekar(9, 10))
# print(isKaprekar(2223, 10))
# print(isKaprekar(22823, 10))
# print(isKaprekar(9, 17))
# print(isKaprekar(225, 17))
# print(isKaprekar(999, 17))
#
#
# # 89
# def horse_position(p, q):
#     """
#     0,0 0,1 0,2 0,3 0,4
#     1,0 1,1 1,2 1,3 1,4
#     2,0 2,1 [2,2] 2,3 2,4
#     3,0 3,1 3,2 3,3 3,4
#     4,0 4,1 4,2 4,3 4,4
#     """
#     neighbors = [[p - 2 , q - 1], [p - 2, q + 1], [p - 1 , q - 2], [p - 1, q + 2],
#                  [p + 1, q - 2], [p + 1, q + 2], [p + 2, q - 1], [p + 2, q + 1]]
#     return neighbors
#
#
#
# def can_reach_all_positions(p, q, mat, num_pos, visited):
#     left = 2
#     up = 2
#     down = mat.shape[0] - 3
#     right = mat.shape[1] - 3
#
#     # while True:
#     positions = horse_position(p, q)
#     # mat[p, q] = 1
#     visited.add((p, q))
#
#     positions = [elem for elem in positions if mat[elem[0], elem[1]] != -1 and (elem[0], elem[1]) not in visited]
#     if len(positions) == 0 and num_pos != (mat.shape[0] - 4) * (mat.shape[1] - 4):
#         continue
#     # return False
#     for position in positions:
#         if num_pos == (mat.shape[0] - 4) * (mat.shape[1] - 4):
#             return True
#         else:
#             visited.add((position[0], position[1]))
#             return can_reach_all_positions(position[0], position[1], mat, num_pos + 1, visited)
#         # visited.remove((position[0], position[1]))
#
#             # visited.remove((position[0], position[1]))
#
#     # return False
#
# TODO FIX
# def knightTour(w, h):
#     mat = np.zeros((w + 4, h + 4))
#     mat[:, :2] = -1.0
#     mat[:2, :] = -1.0
#     mat[-2:, :] = -1.0
#     mat[:, -2:] = -1.0
#
#     num_all_reach_positions = 0
#
#     for i in range(2, w + 2):
#         for j in range(2, h + 2):
#             num_visited_positions = 1
#             new_mat = deepcopy(mat)
#             visited = set()
#             if can_reach_all_positions(i, j, new_mat, num_visited_positions, visited):
#                 num_all_reach_positions += 1
#
#     return num_all_reach_positions
#
#
# print(knightTour(6, 6))
# print(knightTour(5, 6))
#
# print(knightTour(4, 6))
# print(knightTour(7, 3))
#
# print(knightTour(8, 6))


# 90
from itertools import permutations


# def maxCombine(xs):
#     perms = list(permutations(xs))
#     max_val = -1
#     for perm in perms:
#         res = ""
#         for elem in perm:
#             res += str(elem)
#         if int(res) > max_val:
#             max_val = int(res)
#
#     return max_val
#
# print(maxCombine([1, 3, 3, 4, 55]))
# print(maxCombine([71, 45, 23, 4, 5]))
# print(maxCombine([14, 43, 53, 114, 55]))
# print(maxCombine([1, 34, 3, 98, 9, 76, 45, 4]))
# print(maxCombine([54, 546, 548, 60]))

# 91
#
# def allign(values):
#     perms = list(permutations(values))
#     best_perm = None
#     best_perm_score = -1
#     for perm in perms:
#         score = 0
#         for i in range(len(perm) - 1):
#             if perm[i][-1] == perm[i + 1][0]:
#                 score += 1
#         if score > best_perm_score:
#             best_perm_score = score
#             best_perm = perm
#
#     return best_perm
#
# def findLongestChain(items):
#     longest_chain_length = -1
#     longest_chain = []
#
#     for i in range(2, len(items) + 1):
#         combs = list(combinations(items, i))
#         for comb in combs:
#             # print(comb, "prior")
#             comb = allign(comb)
#             # print(comb, "after")
#             chain = []
#             for j in range(len(comb) - 1):
#                 if len(chain) == 0:
#                     if comb[j][-1] == comb[j + 1][0]:
#                         chain = [comb[j], comb[j + 1]]
#                 else:
#                     if comb[j][0] == chain[-1][-1] and comb[j] not in chain:
#                         chain.append(comb[j])
#             if len(chain) > longest_chain_length:
#                 longest_chain_length = len(chain)
#                 longest_chain = chain
#     return longest_chain
#
#
# print(findLongestChain(["certain", "each", "game", "involves", "starting", "with", "word"]))
# print(findLongestChain(["audino", "bagon", "kangaskhan", "banette", "bidoof", "braviary", "exeggcute", "yamask"]))
# print(findLongestChain(["harp", "poliwrath", "poochyena", "porygon2", "porygonz", "archana"]))
# print(findLongestChain(["scolipede", "elephant", "zeaking", "sealeo", "silcoon", "tigers"]))
# print(findLongestChain(["loudred", "lumineon", "lunatone", "machamp", "magnezone", "nosepass", "petilil", "pidgeotto", "pikachu"]))

# 92
#
# def last_friday_of_month(year, month):
#     for i in range(31, -1, -1):
#         try:
#             d = datetime.date(year, month, i)
#         except ValueError:
#             continue
#         day = d.strftime("%A")
#         if day == "Friday":
#             return i
#
# print(last_friday_of_month(1900, 4))

# 93

# def isLeapYear(year):
#     if year % 4 != 0:
#         return False
#     elif year % 4 == 0:
#         if year % 100 == 0:
#             if year % 400 == 0:
#                 return True
#             else:
#                 return False
#         return True
#
# print(isLeapYear(2017))


# 94
# def lcm(a, b):
#     return (a * b) / gcd(a, b)
#
# def lcm_multiple(items):
#     items = [abs(item) for item in items]
#     res = []
#     combs = list(combinations(items, 2))
#     for comb in combs:
#         res.append(lcm(comb[0], comb[1]))
#     return int(max(res))
#
#
# print(lcm_multiple([2, 4, 8]))
#
# print(lcm_multiple([4, 8, 12]))
#
# print(lcm_multiple([3, 4, 5, 12, 40]))
#
# print(lcm_multiple([11, 33, 90]))
#
# print(lcm_multiple([-50, 25, -45, -18, 90, 447]))


# 95

# def leftFactorial(n):
#     if n == 0:
#         return 0
#     elif n == 1:
#         return 1
#     elif n == 2:
#         return 2
#
#     res = 0
#     for i in range(0, n):
#         # print(i)
#         res += factorial(i)
#
#     return res
#
#
# print(leftFactorial(19))

# 96
#
# from collections import Counter
# def letter_frequency(s):
#     cnt = Counter(s)
#     res = []
#     for (k, v) in cnt.items():
#         res.append([k, v])
#     return res
#
# print(letter_frequency("Not all that Mrs. Bennet, however"))


# 97
# def linearCongGenerator(r0, a, c, m, n):
#     if n == 0:
#         return r0
#     else:
#         return (a * linearCongGenerator(r0, a, c, m, n - 1) + c) % m
#
# print(linearCongGenerator(324, 1145, 177, 2148, 3))
# print(linearCongGenerator(234, 11245, 145, 83648, 4))
# print(linearCongGenerator(85, 11, 1234, 214748, 5))
# print(linearCongGenerator(0, 1103515245, 12345, 2147483648, 1))
# print(linearCongGenerator(0, 1103515245, 12345, 2147483648, 2))


# 98

# TODO large number multiplication
def mult(s1, s2):
    pass


print(mult("18446744073709551616", "18446744073709551616"))
print(mult("31844674073709551616", "1844674407309551616"))
print(mult("1846744073709551616", "44844644073709551616"))
print(mult("1844674407370951616", "1844674407709551616"))
print(mult("2844674407370951616", "1844674407370955616"))

# 99
# def lcs(a, b):
#     c = [[0 for _ in range(len(b))] for _ in range(len(a))]
#     for i in range(len(a)):
#         for j in range(len(b)):
#             if a[i] == b[j]:
#                 c[i][j] = 1
#             else:
#                 c[i][j] = 0
#
#     #     for i in range(len(c)):
#     #         for j in range(len(c[i])):
#     #             print(c[i][j], end=" ")
#     #         print()
#
#     d = []
#     for i in range(len(c)):
#         for j in range(len(c[i])):
#             if c[i][j] == 1:
#                 d.append((i, j))
#
#     # d = sorted(d, key=lambda x: x[0])
#     # d = sorted(d, key=lambda x: x[1])
#
#     res = ""
#
#     max_len = -1
#
#     l_, r_ = d[0][0], d[0][1]
#     cur = a[l_]
#     for i in range(1, len(d)):
#         l, r = d[i][0], d[i][1]
#         if l > l_ and r > r_:
#             l_ = l
#             r_ = r
#             cur += a[l_]
#
#     if len(cur) > max_len:
#         max_len = len(cur)
#         res = cur
#
#     d = sorted(d, key=lambda x: x[0])
#
#     l_, r_ = d[0][0], d[0][1]
#     cur = a[l_]
#     for i in range(1, len(d)):
#         l, r = d[i][0], d[i][1]
#         if l > l_ and r > r_:
#             l_ = l
#             r_ = r
#             cur += a[l_]
#
#     if len(cur) > max_len:
#         max_len = len(cur)
#         res = cur
#
#     d = sorted(d, key=lambda x: x[1])
#
#     l_, r_ = d[0][0], d[0][1]
#     cur = a[l_]
#     for i in range(1, len(d)):
#         l, r = d[i][0], d[i][1]
#         if l > l_ and r > r_:
#             l_ = l
#             r_ = r
#             cur += a[l_]
#
#     if len(cur) > max_len:
#         max_len = len(cur)
#         res = cur
#
#     d = sorted(d, key=lambda x: x[0] + x[1])
#
#     l_, r_ = d[0][0], d[0][1]
#     cur = a[l_]
#     for i in range(1, len(d)):
#         l, r = d[i][0], d[i][1]
#         if l > l_ and r > r_:
#             l_ = l
#             r_ = r
#             cur += a[l_]
#
#     if len(cur) > max_len:
#         max_len = len(cur)
#         res = cur
#
#
#     return res
#
#
# print(lcs("thisisatest", "testing123testing"))
# print(lcs("ABCDGH", "AEDFHR"))
# print(lcs("AGGTAB", "GXTXAYB"))
# print(lcs("BDACDB", "BDCB"))
# print(lcs("ABAZDC", "BACBAD"))


# 100

#
# def merge(a, b):
#     i = 0
#     j = 0
#     c = []
#     while i < len(a) and j < len(b):
#         if a[i] < b[j]:
#             c.append(a[i])
#         else:
#             c.append(b[j])
#
#     while j < len(b):
#         c.append(b[j])
#
#     while i < len(a):
#         c.append(a[i])
#
#     return c
#
#
# def longest_increasing_subsequence(vals):
#     lis = []
#     print(vals)
#
#     # sums = []
#     # s = 0
#     # for i in range(len(vals)):
#     #     s += vals[i]
#     #     sums.append(s)
#     # print(sums)
#
#     for i in range(len(vals) - 1, -1, -1):
#         new_vals = vals[i:]
#         ss = [new_vals[0]]
#         ids = [i]
#         for j, val in enumerate(new_vals[1:]):
#             if val >= ss[-1]:
#                 ss.append(val)
#                 ids.append(i + j + 1)
#         print(new_vals, ss, ids)
#
#     print("*" * 10)
#
#     for i in range(len(vals)):
#         new_vals = vals[i:]
#         ss = [new_vals[0]]
#         ids = [i]
#         for j, val in enumerate(new_vals[1:]):
#             if val >= ss[-1]:
#                 ss.append(val)
#                 ids.append(i + j + 1)
#         print(new_vals, ss, ids)
#
#     return new_vals


# print(longest_increasing_subsequence([3, 10, 2, 1, 20]))
# print(longest_increasing_subsequence([2, 7, 3, 5, 8]))
# print(longest_increasing_subsequence([2, 6, 4, 5, 1]))
# print(longest_increasing_subsequence([10, 22, 9, 33, 21, 50, 60, 80]))
# print(longest_increasing_subsequence([0, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15]))

# 101
# def longest_string_challenge(strings):
#     max_len = max([len(string) for string in strings])
#     return [string for string in strings if len(string) == max_len]


# 102
# def look_say_sequence(inp):
#     res = ""
#     digit = None
#     cnt = 0
#     for i, c in enumerate(inp):
#         if digit is None:
#             digit = c
#             cnt = 1
#         else:
#             if digit == c:
#                 cnt += 1
#             else:
#                 res += f"{cnt}{digit}"
#                 cnt = 1
#                 digit = c
#
#     if res[-1] != inp[-1]:
#         res += f"{cnt}{digit}"
#
#     return res
#
# print(look_say_sequence("1211"))


# 103

from collections import Counter
# def word_frequency(text, n):
#     text = text.lower()
#     text = text.split()
#     cnt = Counter(text)
#     res = []
#     for (k, v) in cnt.items():
#         res.append([k, v])
#     res = sorted(res, key=lambda x: (-1) * x[1])
#     return res[:n]
#
#
# print(word_frequency("c d a d c a b d d c", 4))


# 104
from typing import Dict, List, Set, Tuple, Optional, Any, Callable, NoReturn, Union, Mapping, Sequence, Iterable


# def dot_product(*arg):
#     if len(arg) > 2:
#         return None
#
#     a = arg[0]
#     b = arg[1]
#
#     if isinstance(a, List) == False or isinstance(b, List) == False:
#         return None
#
#     for elem in a:
#         if isinstance(elem, float) == False and isinstance(elem, int) == False:
#             return None
#
#     for elem in b:
#         if isinstance(elem, float) == False and isinstance(elem, int) == False:
#             return None
#
#     if len(a) != len(b):
#         return False
#
#     return sum([i * j for (i, j) in zip(a, b)])
#
#
# print(dot_product([3, 2, 1], [2, 4, 2]))


# 105

# TODO figure out types of products: element wise, scalar, dot, cross

# def crossProduct(a, b):
#     if len(a) != len(b):
#         return None
#     res = []
#     if len(a) == 3:
#         res = [a[0] * b[1] - a[1] * b[0], a[2] * b[0] - a[0] * b[2], a[1] * b[2] - a[2] * b[1]]
#     elif len(a) == 2:
#         res = [a[0] * b[1] - a[1] * b[0]]
#     elif len(a) == 1:
#         res = [a[0] * b[0]]
#     return res
#
#
# print(crossProduct([1, 2], [4, 5]))


# 106

# def is_taxicab(n):
#     ways = set()
#     rad = int(math.sqrt(n))
#     for i in range(1, rad - 1):
#         for j in range(i + 1, rad):
#             if i + j == n and is_cube(i) and is_cube(j) and tuple(sorted([i, j])) not in ways:
#                 ways.add(tuple(sorted([i, j])))
#                 # print(i, j)
#                 if len(ways) == 2:
#                     return True
#     return False
#
#
# def taxicab_numbers(n):
#     num = 1
#     k = 0
#     res = []
#     while k < n:
#         if is_taxicab(num):
#             res.append(num)
#             print(num)
#             k += 1
#         num += 1
#     return res

# print(taxicab_numbers(4))
# print(taxicab_numbers(25))
# print(taxicab_numbers(39))
# 107
# TODO solve zig sag
# https://www.freecodecamp.org/learn/coding-interview-prep/rosetta-code/zig-zag-matrix

def zigzag_numbers(n):
    mat = [[0 for _ in range(n)] for _ in range(n)]
    p, q = 0, 1
    k = 1
    while p != n - 1 or q != n - 1:
        mat[p][q] = k
        if p == 0:
            p += 1
            q -= 1
        # if
        k += 1


# 108

def tail(s):
    return s[1:]


# def levenshtein_distance(a, b):
#     if len(b) == 0:
#         return len(a)
#     elif len(a) == 0:
#         return len(b)
#     elif a[0] == b[0]:
#         return levenshtein_distance(tail(a), tail(b))
#     else:
#         return 1 + min(levenshtein_distance(tail(a), b), levenshtein_distance(b, tail(a)), levenshtein_distance(tail(a), tail(b)))
#
#
# print(levenshtein_distance("kitten", "sitting"))


# 109

# test_data = [
#   { "name": 'Tyler Bennett', "id": 'E10297', "salary": 32000, "dept": 'D101' },
#   { "name": 'John Rappl', "id": 'E21437', "salary": 47000, "dept": 'D050' },
#   { "name": 'George Woltman', "id": 'E00127', "salary": 53500, "dept": 'D101' },
#   { "name": 'Adam Smith', "id": 'E63535', "salary": 18000, "dept": 'D202' },
#   { "name": 'Claire Buckman', "id": 'E39876', "salary": 27800, "dept": 'D202' },
#   { "name": 'David McClellan', "id": 'E04242', "salary": 41500, "dept": 'D101' },
#   { "name": 'Rich Holcomb', "id": 'E01234', "salary": 49500, "dept": 'D202' },
#   { "name": 'Nathan Adams', "id": 'E41298', "salary": 21900, "dept": 'D050' },
#   { "name": 'Richard Potter', "id": 'E43128', "salary": 15900, "dept": 'D101' },
#   { "name": 'David Motsinger', "id": 'E27002', "salary": 19250, "dept": 'D202' },
#   { "name": 'Tim Sampair', "id": 'E03033', "salary": 27000, "dept": 'D101' },
#   { "name": 'Kim Arlich', "id": 'E10001', "salary": 57000, "dept": 'D190' },
#   { "name": 'Timothy Grove', "id": 'E16398', "salary": 29900, "dept": 'D190' }
# ]
#
#
# def get_groups(data, groupby):
#     groups = set()
#     for elem in data:
#         groups.add(elem[groupby])
#     return list(groups)
#
#
# def select_data(n, data, group_col, group_val, orderby_column):
#     new_data = []
#     for elem in data:
#         if elem[group_col] == group_val:
#             new_data.append(elem)
#     return sorted(new_data, key=lambda x: (-1) * x[orderby_column])[:n]
#
#
# def top_rank_per_group(n, data, groupby, orderby):
#     res = []
#     groups = get_groups(data, groupby)
#     for group in groups:
#         new_data = select_data(n, data, groupby, group, orderby)
#         # if len(data) != 0:
#         res.append(new_data)
#     return res
#
#
# print(top_rank_per_group(10, test_data, "dept", "salary"))


# 110
# deps = {"des_system_lib": ["std", "synopsys", "std_cell_lib", "dw02", "dw01", "ramlib", "ieee"],
#         "dw01": ["ieee", "dware", "gtech"],
#         "dw02": ["ieee", "dware"],
#         "dw03": ["std", "synopsys", "dware", "dw03", "dw02", "dw01", "ieee", "gtech"],
#         "dw04": ["dw04", "ieee", "dw01", "dware", "gtech"],
#         "dw05": ["dw05", "ieee", "dware"],
#         "dw06": ["dw06", "ieee", "dware"],
#         "dw07": ["ieee", "dware"],
#         "dware": ["ieee"],
#         "gtech": ["ieee"],
#         "ramlib": ["std", "ieee"],
#         "std_cell_lib": ["ieee"],
#         "synopsys": []}
#
#
# def topological_sort(data):
#     interior_degrees = {k: 0 for k in data.keys()}
#     for (node, neighbors) in data.items():
#         for neighbor in neighbors:
#             if neighbor in interior_degrees:
#                 interior_degrees[neighbor] += 1
#             else:
#                 interior_degrees[neighbor] = 0
#     interior_degrees = [(k, v) for (k, v) in interior_degrees.items()]
#     interior_degrees = sorted(interior_degrees, key=lambda x: x[1])
#     min_deg = 1e9
#     for (_, deg) in interior_degrees:
#         if deg < min_deg:
#             min_deg = deg
#
#     res = [elem[0] for elem in interior_degrees if elem[1] == min_deg]
#
#     visited = deepcopy(res)
#
#     while len(visited) != 0:
#         node = visited.pop(0)
#         if node in data:
#             neighbors = data[node]
#         else:
#             continue
#         for neighbor in neighbors:
#             if neighbor not in visited:
#                 visited.append(neighbor)
#                 res.append(neighbor)
#
#     return res
#
# print(topological_sort(deps))

# 111
#
# test_image = [
#     '                               ',
#     '#########       ########       ',
#     '###   ####     ####  ####      ',
#     '###    ###     ###    ###      ',
#     '###   ####     ###             ',
#     '#########      ###             ',
#     '### ####       ###    ###      ',
#     '###  ####  ### ####  #### ###  ',
#     '###   #### ###  ########  ###  ',
#     '                               '
# ]
#
#
# def doubled_hashes(input):
#     for i in range(len(input) - 1):
#         if input[i] == input[i + 1] and input[i] == "#":
#             return True
#     return False
#
#
# def count_transitions(items):
#     res = 0
#     for i in range(len(items) - 1):
#         if items[i] == 0 and items[i + 1] == 1:
#             res += 1
#     return res
#
#
# def count_blacks(items):
#     return len([i for i in items if i == 1])
#
#
# def step_1(mat):
#     coords = [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]
#     a = np.zeros_like(mat)
#     b = np.zeros_like(mat)
#     res = []
#
#     for i in range(1, mat.shape[0] - 1):
#         for j in range(1, mat.shape[1] - 1):
#             submat = mat[i - 1: i + 2, j - 1: j + 2]
#             neighbors = [mat[i + c[0], j + c[1]] for c in coords]
#             if -1 in neighbors:
#                 continue
#             print(neighbors)
#             print(submat.shape)
#             a[i][j] = count_transitions(neighbors)
#             b[i][j] = count_blacks(neighbors)
#
#             if mat[i][j] == 1:
#                 if 2 <= b[i][j] <= 6:
#                     if a[i][j] == 1:
#                         if neighbors[0] == 0 or neighbors[2] == 0 or neighbors[4] == 0:
#                             if neighbors[2] == 0 or neighbors[4] == 0 or neighbors[6] == 0:
#                                 res.append((i, j))
#
#     return res
#
#
# def step_2(mat):
#     coords = [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]
#     a = np.zeros_like(mat)
#     b = np.zeros_like(mat)
#     res = []
#
#     for i in range(1, mat.shape[0] - 1):
#         for j in range(1, mat.shape[1] - 1):
#             submat = mat[i - 1: i + 2, j - 1: j + 2]
#             neighbors = [mat[i + c[0], j + c[1]] for c in coords]
#             if -1 in neighbors:
#                 continue
#             print(neighbors)
#             print(submat.shape)
#             a[i][j] = count_transitions(neighbors)
#             b[i][j] = count_blacks(neighbors)
#
#             if mat[i][j] == 1:
#                 if 2 <= b[i][j] <= 6:
#                     if a[i][j] == 1:
#                         if neighbors[0] == 0 or neighbors[2] == 0 or neighbors[6] == 0:
#                             if neighbors[0] == 0 or neighbors[4] == 0 or neighbors[6] == 0:
#                                res.append((i, j))
#
#     return res
#
#
# def thin_image(img):
#     mat = [[-1 for _ in range(len(img[0]) + 2)] for _ in range(len(img) + 2)]
#     print(mat)
#     for i, row in enumerate(img):
#         for j, chr in enumerate(row):
#             if chr == "#":
#                 mat[i + 1][j + 1] = 1
#             else:
#                 mat[i + 1][j + 1] = 0
#     mat = np.array(mat)
#     print(mat)
#     print(mat.shape)
#     # exit()
#
#     points_1 = step_1(mat)
#     points_2 = step_2(mat)
#
#     points = points_1 + points_2
#
#     for (x, y) in points:
#         mat[x][y] = 0
#     print(mat)
#
#     for i in range(len(mat)):
#         for j in range(len(mat[i])):
#             print(mat[i][j], end=" ")
#         print()
#
#     # for row in img:
#     #     while doubled_hashes()
#
#
# print(thin_image(test_image))

# 112

# def zeckendorf(n, fibos):
#     binary = bin(n)[2:][::-1]
#     res = 0
#     m = len(binary)
#     for b in binary:
#         res += int(b) * fibos[m]
#         m -= 1
#     return binary
#
#
# fibos = [nth_fibo(i) for i in range(1, 10)]
#
# for i in range(0, 20):
#     arr = nth_fibo
#     print(i, zeckendorf(i, fibos))

# 113
# def word_wrap(text, n):
#     res = []
#     while len(text) > 0:
#         res.append(text[:n])
#         text = text[n:]
#     return len(res)
#
# s = "Wrap text using a more sophisticated algorithm such as the Knuth and Plass TeX algorithm. If your language " \
#     "provides this, you get easy extra credit, but you must reference documentation indicating that the algorithm is " \
#     "something better than a simple minimum length algorithm."
#
# print(word_wrap(s, 80))
# print(word_wrap(s, 42))

# 114

# def loopSimult(A):
#     res = ["" for _ in range(len(A[0]))]
#     for i in range(len(A)):
#         for j in range(len(A[i])):
#             res[j] += str(A[i][j])
#         # res.append(A[i][0])
#     return res
#
#
# print(loopSimult([["a", "b", "c", "d"], ["A", "B", "C", "d"], [1, 2, 3, 4]]))
# 115
# ss = [0] * 30
#
# def S(n):
#     global ss
#     if n == 1:
#         return 4
#     else:
#         if ss[n] == 0:
#             ss[n] = S(n - 1) ** 2 - 2
#         return ss[n]
#
#
# def lucas_lehmer(p):
#     global ss
#     exponentiated = pow(2, p) - 1
#     # print(exponentiated, S(p-1))
#     if S(p - 1) % exponentiated == 0:
#         return True
#     return False
#
#
# # for i in range(1, 21):
# #     S(i)
#
# for i in [2, 11, 13, 15, 17, 19, 21]:
#     print(i, lucas_lehmer(i))

# 116

# def luhnTest(code):
#     code = code[::-1]
#     s1 = 0
#
#     evens = []
#
#     for i, c in enumerate(code):
#         if i % 2 == 0:
#             s1 += int(c)
#         else:
#             evens.append(int(c))
#
#     evens = [2 * num for num in evens]
#
#     evens = [sum_digits(num) for num in evens]
#
#     s2 = sum(evens)
#
#     res = s1 + s2
#
#     if res % 10 == 0:
#         return True
#     return False
#
#
# print(luhnTest("4111111111111111"))
# print(luhnTest("4111111111111112"))
# print(luhnTest("49927398716"))
# print(luhnTest("49927398717"))
# print(luhnTest("1234567812345678"))
# print(luhnTest("1234567812345670"))

# 117
# def is_palindrome(n):
#     return str(n) == str(n)[::-1]
#
#
# def lychrel(n):
#     iters = 500
#     for i in range(iters):
#         aux = n + int(str(n)[::-1])
#         if is_palindrome(aux):
#             return False
#         n = aux
#     return True
#
#
# for i in [12, 55, 196, 879, 44987, 7059]:
#     print(lychrel(i))


# 118
# TODO https://www.freecodecamp.org/learn/coding-interview-prep/rosetta-code/lzw-compression

def LZW(compress_data, input):
    if compress_data is True and isinstance(input, str) is False:
        raise Exception("Wrong!")
    if compress_data is False and isinstance(input, List) is False:
        raise Exception("Wrong!")

    asciis = [chr(i) for i in range(128)]
    #
    # if compress_data is True:
    #     res = [asciis.index(letter) for letter in input]
    # else:
    #     res = [chr(num) for num in input]

    table = set()
    if compress_data is True:
        p = input[0]
        for c in input[1:]:
            if p + c in table:
                p = p + c
            else:
                res.append(p)
            table.add(p + c)
            p = c
    else:
        old = input
        for new in input[1:]:
            if new not in table:
                s = translation(old)
                s = s + c
            else:
                s = translation(new)
            res.append(s)
            c = s[0]
            table.add(old + c)
            old = new
    return res

# print(LZW(True, "TOBEORNOTTOBEORTOBEORNOT"))
# print(LZW(False, [84, 79, 66, 69, 79, 82, 78, 79, 84, 256, 258, 260, 265, 259, 261, 263]))
# print(LZW(True, "TOBEORNOTTOBEORTOBEORNOT"))
# print(LZW(False, [84, 79, 66, 69, 79, 82, 78, 79, 84, 256, 258, 260, 265, 259, 261, 263]))
# print(LZW(True, "0123456789"))
# print(LZW(False, [48, 49, 50, 51, 52, 53, 54, 55, 56, 57]))
# print(LZW(True, "BABAABAAA"))
# print(LZW(False, [66, 65, 256, 257, 65, 260]))


# 119
# todo https://www.freecodecamp.org/learn/coding-interview-prep/rosetta-code/knapsack-problem0-1
# https://www.freecodecamp.org/learn/coding-interview-prep/rosetta-code/knapsack-problemunbounded
def knaspackUnbounded(items, maxweight, maxvolume):
    pass


print(knapsackUnbounded([{ name:"panacea", value:3000, weight:0.3, volume:0.025 }, { name:"ichor", value:1800, weight:0.2, volume:0.015 }, { name:"gold", value:2500, weight:2, volume:0.002 }], 25, 0.25))
print(knapsackUnbounded([{ name:"panacea", value:3000, weight:0.3, volume:0.025 }, { name:"ichor", value:1800, weight:0.2, volume:0.015 }, { name:"gold", value:2500, weight:2, volume:0.002 }], 55, 0.25))
print(knapsackUnbounded([{ name:"panacea", value:3000, weight:0.3, volume:0.025 }, { name:"ichor", value:1800, weight:0.2, volume:0.015 }, { name:"gold", value:2500, weight:2, volume:0.002 }], 25, 0.15))
print(knapsackUnbounded([{ name:"panacea", value:3000, weight:0.3, volume:0.025 }, { name:"ichor", value:1800, weight:0.2, volume:0.015 }, { name:"gold", value:2500, weight:2, volume:0.002 }], 35, 0.35))
print(knapsackUnbounded([{ name:"panacea", value:3000, weight:0.3, volume:0.025 }, { name:"ichor", value:1800, weight:0.2, volume:0.015 }, { name:"gold", value:2500, weight:2, volume:0.002 }], 15, 0.25))


# 120
def kdNN(fpoints, fpoint):
    pass


print(kdNN([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]], [9, 2]))
print(kdNN([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]], [7, 1]))
print(kdNN([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]], [3, 2]))
print(kdNN([[2, 3, 1], [9, 4, 5], [4, 6, 7], [1, 2, 5], [7, 8, 9], [3, 6, 1]], [1, 2, 3]))
print(kdNN([[2, 3, 1], [9, 4, 5], [4, 6, 7], [1, 2, 5], [7, 8, 9], [3, 6, 1]], [4, 5, 6]))
print(kdNN([[2, 3, 1], [9, 4, 5], [4, 6, 7], [1, 2, 5], [7, 8, 9], [3, 6, 1]], [8, 8, 8]))

# 121

def knapsack(items, maxweight):
    pass

print(knapsack([{ name:'map', weight:9, value:150 }, { name:'compass', weight:13, value:35 },
                { name:'water', weight:153, value:200 }, { name:'sandwich', weight:50, value:160 }, { name:'glucose', weight:15, value:60 },
                { name:'tin', weight:68, value:45 }, { name:'banana', weight:27, value:60 }, { name:'apple', weight:39, value:40 }], 100))
print(knapsack([{ name:'map', weight:9, value:150 }, { name:'compass', weight:13, value:35 }, { name:'water', weight:153, value:200 },
                { name:'sandwich', weight:50, value:160 }, { name:'glucose', weight:15, value:60 }, { name:'tin', weight:68, value:45 },
                { name:'banana', weight:27, value:60 }, { name:'apple', weight:39, value:40 }], 200))
print(knapsack([{ name:'cheese', weight:23, value:30 }, { name:'beer', weight:52, value:10 }, { name:'suntan cream', weight:11, value:70 },
                { name:'camera', weight:32, value:30 }, { name:'T-shirt', weight:24, value:15 }, { name:'trousers', weight:48, value:10 },
                { name:'umbrella', weight:73, value:40 }], 100))
print(knapsack([{ name:'cheese', weight:23, value:30 }, { name:'beer', weight:52, value:10 }, { name:'suntan cream', weight:11, value:70 },
                { name:'camera', weight:32, value:30 }, { name:'T-shirt', weight:24, value:15 }, { name:'trousers', weight:48, value:10 },
                { name:'umbrella', weight:73, value:40 }], 200))
print(knapsack([{ name:'waterproof trousers', weight:42, value:70 }, { name:'waterproof overclothes', weight:43, value:75 },
                { name:'note-case', weight:22, value:80 }, { name:'sunglasses', weight:7, value:20 }, { name:'towel', weight:18, value:12 },
                { name:'socks', weight:4, value:50 }, { name:'book', weight:30, value:10 }], 100))
print(knapsack([{ name:'waterproof trousers', weight:42, value:70 }, { name:'waterproof overclothes', weight:43, value:75 },
                { name:'note-case', weight:22, value:80 }, { name:'sunglasses', weight:7, value:20 }, { name:'towel', weight:18, value:12 },
                { name:'socks', weight:4, value:50 }, { name:'book', weight:30, value:10 }], 200))


# 122
# todo https://www.freecodecamp.org/learn/coding-interview-prep/rosetta-code/knapsack-problembounded

def findBestPack(data, maxweight):
    pass


print(findBestPack([{ name:'map', weight:9, value:150, pieces:1 }, { name:'compass', weight:13, value:35, pieces:1 }, { name:'water', weight:153, value:200, pieces:2 }, { name:'sandwich', weight:50, value:60, pieces:2 }, { name:'glucose', weight:15, value:60, pieces:2 }, { name:'tin', weight:68, value:45, pieces:3 }, { name:'banana', weight:27, value:60, pieces:3 }, { name:'apple', weight:39, value:40, pieces:3 }, { name:'cheese', weight:23, value:30, pieces:1 },
                    { name:'beer', weight:52, value:10, pieces:3 }, { name:'suntan, cream', weight:11, value:70, pieces:1 },
                    { name:'camera', weight:32, value:30, pieces:1 }, { name:'T-shirt', weight:24, value:15, pieces:2 }], 300))

print(findBestPack([{ name:'map', weight:9, value:150, pieces:1 }, { name:'compass', weight:13, value:35, pieces:1 },
                    { name:'water', weight:153, value:200, pieces:2 }, { name:'sandwich', weight:50, value:60, pieces:2 }, { name:'glucose', weight:15, value:60, pieces:2 }, { name:'tin', weight:68, value:45, pieces:3 }, { name:'banana', weight:27, value:60, pieces:3 }, { name:'apple', weight:39, value:40, pieces:3 }, { name:'cheese', weight:23, value:30, pieces:1 }, { name:'beer', weight:52, value:10, pieces:3 }, { name:'suntan, cream', weight:11, value:70, pieces:1 },
                    { name:'camera', weight:32, value:30, pieces:1 }, { name:'T-shirt', weight:24, value:15, pieces:2 }], 400))

print(findBestPack([{ name:'map', weight:9, value:150, pieces:1 }, { name:'compass', weight:13, value:35, pieces:1 },
                    { name:'water', weight:153, value:200, pieces:2 }, { name:'sandwich', weight:50, value:60, pieces:2 },
                    { name:'glucose', weight:15, value:60, pieces:2 }, { name:'tin', weight:68, value:45, pieces:3 },
                    { name:'banana', weight:27, value:60, pieces:3 }, { name:'apple', weight:39, value:40, pieces:3 },
                    { name:'cheese', weight:23, value:30, pieces:1 }, { name:'beer', weight:52, value:10, pieces:3 },
                    { name:'suntan, cream', weight:11, value:70, pieces:1 }, { name:'camera', weight:32, value:30, pieces:1 },
                    { name:'T-shirt', weight:24, value:15, pieces:2 }], 500))

print(findBestPack([{ name:'map', weight:9, value:150, pieces:1 }, { name:'compass', weight:13, value:35, pieces:1 },
                    { name:'water', weight:153, value:200, pieces:2 }, { name:'sandwich', weight:50, value:60, pieces:2 },
                    { name:'glucose', weight:15, value:60, pieces:2 }, { name:'tin', weight:68, value:45, pieces:3 },
                    { name:'banana', weight:27, value:60, pieces:3 }, { name:'apple', weight:39, value:40, pieces:3 },
                    { name:'cheese', weight:23, value:30, pieces:1 }, { name:'beer', weight:52, value:10, pieces:3 },
                    { name:'suntan, cream', weight:11, value:70, pieces:1 }, { name:'camera', weight:32, value:30, pieces:1 },
                    { name:'T-shirt', weight:24, value:15, pieces:2 }], 600))
print(findBestPack([{ name:'map', weight:9, value:150, pieces:1 }, { name:'compass', weight:13, value:35, pieces:1 },
                    { name:'water', weight:153, value:200, pieces:2 }, { name:'sandwich', weight:50, value:60, pieces:2 },
                    { name:'glucose', weight:15, value:60, pieces:2 }, { name:'tin', weight:68, value:45, pieces:3 },
                    { name:'banana', weight:27, value:60, pieces:3 }, { name:'apple', weight:39, value:40, pieces:3 },
                    { name:'cheese', weight:23, value:30, pieces:1 }, { name:'beer', weight:52, value:10, pieces:3 },
                    { name:'suntan, cream', weight:11, value:70, pieces:1 }, { name:'camera', weight:32, value:30, pieces:1 },
                    { name:'T-shirt', weight:24, value:15, pieces:2 }], 700))


# 123
# TODO https://www.freecodecamp.org/learn/coding-interview-prep/rosetta-code/knapsack-problemcontinuous

def knapContinuous(items, maxweight):
    pass

print(knapContinuous([{ "weight":3.8, "value":36, name:"beef" }, { "weight":5.4, "value":43, name:"pork" }, { "weight":3.6, "value":90, name:"ham" }, { "weight":2.4, "value":45, name:"greaves" }, { "weight":4.0, "value":30, name:"flitch" }, { "weight":2.5, "value":56, name:"brawn" }, { "weight":3.7, "value":67, name:"welt" }, { "weight":3.0, "value":95, name:"salami" }, { "weight":5.9, "value":98, name:"sausage" }], 10))
print(knapContinuous([{ "weight":3.8, "value":36, name:"beef" }, { "weight":5.4, "value":43, name:"pork" }, { "weight":3.6, "value":90, name:"ham" }, { "weight":2.4, "value":45, name:"greaves" }, { "weight":4.0, "value":30, name:"flitch" }, { "weight":2.5, "value":56, name:"brawn" }, { "weight":3.7, "value":67, name:"welt" }, { "weight":3.0, "value":95, name:"salami" }, { "weight":5.9, "value":98, name:"sausage" }], 12))
print(knapContinuous([{ "weight":3.8, "value":36, name:"beef" }, { "weight":5.4, "value":43, name:"pork" }, { "weight":3.6, "value":90, name:"ham" }, { "weight":2.4, "value":45, name:"greaves" }, { "weight":4.0, "value":30, name:"flitch" }, { "weight":2.5, "value":56, name:"brawn" }, { "weight":3.7, "value":67, name:"welt" }, { "weight":3.0, "value":95, name:"salami" }, { "weight":5.9, "value":98, name:"sausage" }], 15))
print(knapContinuous([{ "weight":3.8, "value":36, name:"beef" }, { "weight":5.4, "value":43, name:"pork" }, { "weight":3.6, "value":90, name:"ham" }, { "weight":2.4, "value":45, name:"greaves" }, { "weight":4.0, "value":30, name:"flitch" }, { "weight":2.5, "value":56, name:"brawn" }, { "weight":3.7, "value":67, name:"welt" }, { "weight":3.0, "value":95, name:"salami" }, { "weight":5.9, "value":98, name:"sausage" }], 22))
print(knapContinuous([{ "weight":3.8, "value":36, name:"beef" }, { "weight":5.4, "value":43, name:"pork" }, { "weight":3.6, "value":90, name:"ham" }, { "weight":2.4, "value":45, name:"greaves" }, { "weight":4.0, "value":30, name:"flitch" }, { "weight":2.5, "value":56, name:"brawn" }, { "weight":3.7, "value":67, name:"welt" }, { "weight":3.0, "value":95, name:"salami" }, { "weight":5.9, "value":98, name:"sausage" }], 24))


