We provide the preprocessed HIN HeteroGraphine and NELL.

Each directory is a dataset:
-"graph_text.txt" contains all the edges. Each line is an edge (a1 : v1, e, a2 : v2) split by "\t" , where a1 is the type of node 1, v1 is the name of node1, e is the edge type, and a2 and v2 are the node type and node name of node 2.
-"graph_text_train.txt" are the training positive edges for link prediction. Each line is two nodes (a1 : v1, a2 : v2) split by "\t".
-"graph_text_test.txt" are the test positive edges for link prediction. Each line is two nodes (a1 : v1, a2 : v2) split by "\t".

Other files are our preprocessed data to assist the meta-path generation process.

