#!/user/bin/env python
# encoding=utf-8
# @Author: daping
# @Date: 2024/8/4 18:57

import re


class CnDigitsNorm(object):
    ''' 数字标准化 '''
    def __init__(self):
        self.common_used_numerals_tmp = {u'零': 0, u'一': 1, u'二': 2, u'两': 2, u'三': 3, u'四': 4, u'五': 5, u'六': 6, '七': 7,
                                         '八': 8, '九': 9, '十': 10, '百': 100, '千': 1000, '万': 10000, '亿': 100000000,
                                         '壹': 1, '贰': 2, '叁': 3, '肆': 4, '伍': 5, '陆': 6, '柒': 7, '捌': 8, '玖': 9, '貮': 2,
                                         '两': 2, }
        self.common_used_numerals = {}
        for key in self.common_used_numerals_tmp:
            self.common_used_numerals[key] = self.common_used_numerals_tmp[key]
        self.num_str_start_symbol = ['一', '二', '两', '三', '四', '五', '六', '七', '八', '九', '十']
        self.more_num_str_symbol = ['零', '一', '二', '两', '三', '四', '五', '六', '七', '八', '九', '十', '百', '千', '万', '亿', '壹',
                                    '贰', '叁', '肆', '伍', '陆', '柒', '捌', '玖', '貮', '两']

    def chinese2digits(self, uchars_chinese):
        total = 0
        r = 1  # 表示单位：个十百千...
        # corrections: 八一 -> 81 !-> 9
        if all([1 if not x in uchars_chinese else 0 for x in
                ['千亿', '百亿', '十亿', '亿', '千万', '百万', '十万', '万', '千', '百', '十']]):
            total = int(''.join([str(self.common_used_numerals_tmp[char]) for char in uchars_chinese]))
        else:
            for i in range(len(uchars_chinese) - 1, -1, -1):
                val = self.common_used_numerals.get(uchars_chinese[i])
                if val >= 10 and i == 0:  # 应对 十三 十四 十*之类
                    if val > r:
                        r = val
                        total = total + val
                    else:
                        r = r * val
                        # total =total + r * x
                elif val >= 10:
                    r = val if val > r else r * val
                else:
                    total = total + r * val
        return total

    def change_chinese_num2arab(self, ori_str):
        len_str = len(ori_str)
        apro_str = ''
        if len_str == 0:
            return apro_str
        has_num_start = False
        number_str = ''
        for idx in range(len_str):
            if ori_str[idx] in self.num_str_start_symbol:
                if not has_num_start:
                    has_num_start = True
                number_str += ori_str[idx]
            else:
                if has_num_start:
                    if ori_str[idx] in self.more_num_str_symbol:
                        number_str += ori_str[idx]
                        continue
                    else:
                        num_result = str(self.chinese2digits(number_str))
                        number_str = ''
                        has_num_start = False
                        apro_str += num_result
                apro_str += ori_str[idx]
                pass
        if len(number_str) > 0:
            result_num = self.chinese2digits(number_str)
            apro_str += str(result_num)
        return apro_str

    def get_rid_of_punc(self, inputs):
        """
        [usage]: 去除文本中的标点符号，只提取中文英文数字
        """
        cn_text = re.compile(r'[\u4e00-\u9fa5_a-zA-Z0-9]{2,50}', re.IGNORECASE)
        tokens = cn_text.findall(inputs)
        return ''.join(tokens)

    def str_full_width2half_width(self, ustring, remove_blank=True):
        ss = []
        for s in ustring:
            rstring = ""
            for uchar in s:
                inside_code = ord(uchar)
                if inside_code == 12288:  # 全角空格直接转换
                    inside_code = 32
                elif 65281 <= inside_code <= 65374:  # 全角字符（除空格）根据关系转化
                    inside_code -= 65248
                rstring += chr(inside_code)
            ss.append(rstring)
        if not remove_blank:
            return ''.join(ss)
        else:
            return ''.join(''.join(ss).strip().split())