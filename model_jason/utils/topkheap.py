#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# This is a top k heap
#
# @author: Jason Wu (Jasonwbw@yahoo.com)

import heapq


class TopkHeap(object):

    '''Heap save the top k element

    Use the heapq in python.

    Attributes:
            k : top k
            data: a list contain the top k data
    '''

    def __init__(self, k=50):
        self.k = k
        self.data = []

    def push(self, elem):
        '''Push new elem to heap

        Args:
                elem ：the elem to add

        Returns:
                if the elem have added to queue
        '''
        if len(self.data) < self.k:
            heapq.heappush(self.data, elem)
            return True
        else:
            topk_small = self.data[0]
            if elem > topk_small:
                heapq.heapreplace(self.data, elem)
                return True
            return False

    def topk(self):
        '''Get top k elements

        Returns:
                a list of top k
        '''
        return sorted(self.data)[::-1]
