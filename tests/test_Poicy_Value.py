#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
import unittest
import doctest
import Policy_Value
 
# unit tests.
 
 
def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(Policy_Value))
    return tests

# doctests. 
def test_cointoss.py():
    continue


if __name__ == '__main__':
    unittest.main()