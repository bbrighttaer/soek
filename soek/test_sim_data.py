import unittest

from soek import DataNode


class DataNodeTestCase(unittest.TestCase):
    def test_something(self):
        sim_data = DataNode(label='node_label')
        nodes_list = []
        sim_data.data = nodes_list
        data_node = DataNode(label="seed_0")
        nodes_list.append(data_node)

        # sub-nodes of sim data resource
        loss_lst = []
        train_loss_node = DataNode(label="training_loss", data=loss_lst)
        metrics_dict = {}
        metrics_node = DataNode(label="validation_metrics", data=metrics_dict)
        scores_lst = []
        scores_node = DataNode(label="validation_score", data=scores_lst)

        # add sim data nodes to parent node
        sim_data_node = data_node
        if sim_data_node:
            sim_data_node.data = [train_loss_node, metrics_node, scores_node]
        sim_data.to_json(path="./")
        self.assertIsNotNone(sim_data.to_json_str())


if __name__ == '__main__':
    unittest.main()
