# Author: bbrighttaer
# Project: soek
# Date: 7/9/19
# Time: 1:03 PM
# File: sim_data.py


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import os


def _to_dict(data):
    """Converts the data of a ::class::DataNode object into a Python dictionary object."""
    # if it is a literal or leave node (value) then no further processing is needed (base condition)
    if not (isinstance(data, DataNode) or isinstance(data, list) or isinstance(data, set)
            or isinstance(data, dict)):
        return data

    # if it is a ::class::DataNode then it should be processed further
    elif isinstance(data, DataNode):
        return {data.label: _to_dict(data.data)}

    # process list and set items
    elif isinstance(data, list) or isinstance(data, set):
        return [_to_dict(d) for d in data]

    # process dict items
    elif isinstance(data, dict):
        return {k: _to_dict(data[k]) for k in data}


class DataNode(object):
    """Gathers simulation data in a resource tree for later analysis."""

    def __init__(self, label, data=None, metadata=None):
        """
        Creates a ::class::DataNode object for storing data.

        Arguments
        ----------
        :param label: String,
            The name of the node.
        :param data: object,
            Any python object.
        """
        self.label = label
        self.data = data
        self.metadata = metadata

    def to_json(self, path="./"):
        """
        Converts the data node object into a JSON file and persists it in the given directory.

        Argument
        ---------
        :param path: string
            The directory to persist the JSON file.
        """
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, self.label + ".json"), "w") as f:
            json.dump({self.label: _to_dict(self.data), 'metadata': self.metadata}, f)

    def to_json_str(self):
        """Converts the data node to JSON and returns it as a string."""
        return json.dumps({self.label: _to_dict(self.data), 'metadata': self.metadata})
