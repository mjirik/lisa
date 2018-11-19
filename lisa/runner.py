# /usr/bin/env python
# -*- coding: utf-8 -*-
"""
List of functions to run
"""
import logging
logger = logging.getLogger(__name__)


class Runner:
    """
    Create list of functions to run. It can parse a list of functions names
    """
    def __init__(self, main_object=None):
        self.run_list = []
        self.main_object = main_object

        pass

    def __get_function_args_from_input(self, fcn, args, kwargs):
        if type(fcn) in (tuple, list):
            fcn, args, kwargs = fcn
        return fcn, args, kwargs

    def __get_function_from_input_string(self, fcn):

        if type(fcn) is str:
            if self.main_object is not None and hasattr(self.main_object, fcn):
                fcn = getattr(self.main_object, fcn)

                pass
            else:
                raise ValueError("Function '{}' not found in given object.".format(fcn))
        return fcn

    def append(self, fcn, *args, **kwargs):
        """

        :param fcn: function or string name of function or list with [fcn, args, kwargs]
        :param args:
        :param kwargs:
        :return:
        """
        fcn, args, kwargs = self.__get_function_args_from_input(fcn, args, kwargs)
        fcn = self.__get_function_from_input_string(fcn)
        self.run_list.append((fcn, args, kwargs))

    def insert(self, i, fcn, *args, **kwargs):
        """

        :param fcn: function or string name of function or list with [fcn, args, kwargs]
        :param args:
        :param kwargs:
        :return:
        """
        fcn, args, kwargs = self.__get_function_args_from_input(fcn, args, kwargs)
        fcn = self.__get_function_from_input_string(fcn)
        self.run_list.insert(i, (fcn, args, kwargs))

    def extend(self, functions):
        """

        :param functions: list of function or string name of function or list with [fcn, args, kwargs]
        ["function1", "function2", ["function3", [arg1, arg2], {"kwarg1":5, "kwarg2": None}, "function3"]
        :return:
        """
        for fcn in functions:
            self.append(fcn)

    def run(self):
        for fcn_with_params in self.run_list:
            fcn, args, kwargs = fcn_with_params
            fcn(*args, **kwargs)
