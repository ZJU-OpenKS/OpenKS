import os
import os.path as osp
import pdb
import shutil
import copy
from copy import deepcopy
import numpy as np
from itertools import repeat, product

import torch
from scipy.linalg import expm
from torch_geometric.utils import add_self_loops, is_undirected, to_dense_adj
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.io import read_tu_data
from torch_sparse import SparseTensor, coalesce
from torch_scatter import scatter_add
import torch_geometric.transforms as T


class GDC(object):
    r"""Processes the graph via Graph Diffusion Convolution (GDC) from the
    `"Diffusion Improves Graph Learning" <https://www.kdd.in.tum.de/gdc>`_
    paper.

    .. note::

        The paper offers additional advice on how to choose the
        hyperparameters.
        For an example of using GCN with GDC, see `examples/gcn.py
        <https://github.com/rusty1s/pytorch_geometric/blob/master/examples/
        gcn.py>`_.

    Args:
        self_loop_weight (float, optional): Weight of the added self-loop.
            Set to :obj:`None` to add no self-loops. (default: :obj:`1`)
        normalization_in (str, optional): Normalization of the transition
            matrix on the original (input) graph. Possible values:
            :obj:`"sym"`, :obj:`"col"`, and :obj:`"row"`.
            See :func:`GDC.transition_matrix` for details.
            (default: :obj:`"sym"`)
        normalization_out (str, optional): Normalization of the transition
            matrix on the transformed GDC (output) graph. Possible values:
            :obj:`"sym"`, :obj:`"col"`, :obj:`"row"`, and :obj:`None`.
            See :func:`GDC.transition_matrix` for details.
            (default: :obj:`"col"`)
        diffusion_kwargs (dict, optional): Dictionary containing the parameters
            for diffusion.
            `method` specifies the diffusion method (:obj:`"ppr"`,
            :obj:`"heat"` or :obj:`"coeff"`).
            Each diffusion method requires different additional parameters.
            See :func:`GDC.diffusion_matrix_exact` or
            :func:`GDC.diffusion_matrix_approx` for details.
            (default: :obj:`dict(method='ppr', alpha=0.15)`)
        sparsification_kwargs (dict, optional): Dictionary containing the
            parameters for sparsification.
            `method` specifies the sparsification method (:obj:`"threshold"` or
            :obj:`"topk"`).
            Each sparsification method requires different additional
            parameters.
            See :func:`GDC.sparsify_dense` for details.
            (default: :obj:`dict(method='threshold', avg_degree=64)`)
        exact (bool, optional): Whether to exactly calculate the diffusion
            matrix.
            Note that the exact variants are not scalable.
            They densify the adjacency matrix and calculate either its inverse
            or its matrix exponential.
            However, the approximate variants do not support edge weights and
            currently only personalized PageRank and sparsification by
            threshold are implemented as fast, approximate versions.
            (default: :obj:`True`)

    :rtype: :class:`torch_geometric.data.Data`
    """
    def __init__(self, self_loop_weight=1, normalization_in='sym',
                 normalization_out='col',
                 diffusion_kwargs=dict(method='ppr', alpha=0.15),
                 sparsification_kwargs=dict(method='threshold',
                                            avg_degree=64), exact=True):

        self.__calc_ppr__ = get_calc_ppr()

        self.self_loop_weight = self_loop_weight
        self.normalization_in = normalization_in
        self.normalization_out = normalization_out
        self.diffusion_kwargs = diffusion_kwargs
        self.sparsification_kwargs = sparsification_kwargs
        self.exact = exact

        if self_loop_weight:
            assert exact or self_loop_weight == 1

    @torch.no_grad()
    def __call__(self, data):
        N = data.num_nodes
        edge_index = data.edge_index
        if data.edge_attr is None:
            edge_weight = torch.ones(edge_index.size(1),
                                     device=edge_index.device)
        else:
            edge_weight = data.edge_attr
            assert self.exact
            assert edge_weight.dim() == 1

        if self.self_loop_weight:
            edge_index, edge_weight = add_self_loops(
                edge_index, edge_weight, fill_value=self.self_loop_weight,
                num_nodes=N)

        edge_index, edge_weight = coalesce(edge_index, edge_weight, N, N)

        if self.exact:
            edge_index, edge_weight = self.transition_matrix(
                edge_index, edge_weight, N, self.normalization_in)
            diff_mat = self.diffusion_matrix_exact(edge_index, edge_weight, N,
                                                   **self.diffusion_kwargs)
            edge_index, edge_weight = self.sparsify_dense(
                diff_mat, **self.sparsification_kwargs)
        else:
            edge_index, edge_weight = self.diffusion_matrix_approx(
                edge_index, edge_weight, N, self.normalization_in,
                **self.diffusion_kwargs)
            edge_index, edge_weight = self.sparsify_sparse(
                edge_index, edge_weight, N, **self.sparsification_kwargs)

        edge_index, edge_weight = coalesce(edge_index, edge_weight, N, N)
        edge_index, edge_weight = self.transition_matrix(
            edge_index, edge_weight, N, self.normalization_out)

        data.edge_index = edge_index
        data.edge_attr = edge_weight

        return data

    def transition_matrix(self, edge_index, edge_weight, num_nodes,
                          normalization):
        r"""Calculate the approximate, sparse diffusion on a given sparse
        matrix.

        Args:
            edge_index (LongTensor): The edge indices.
            edge_weight (Tensor): One-dimensional edge weights.
            num_nodes (int): Number of nodes.
            normalization (str): Normalization scheme:

                1. :obj:`"sym"`: Symmetric normalization
                   :math:`\mathbf{T} = \mathbf{D}^{-1/2} \mathbf{A}
                   \mathbf{D}^{-1/2}`.
                2. :obj:`"col"`: Column-wise normalization
                   :math:`\mathbf{T} = \mathbf{A} \mathbf{D}^{-1}`.
                3. :obj:`"row"`: Row-wise normalization
                   :math:`\mathbf{T} = \mathbf{D}^{-1} \mathbf{A}`.
                4. :obj:`None`: No normalization.

        :rtype: (:class:`LongTensor`, :class:`Tensor`)
        """
        if normalization == 'sym':
            row, col = edge_index
            deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        elif normalization == 'col':
            _, col = edge_index
            deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
            deg_inv = 1. / deg
            deg_inv[deg_inv == float('inf')] = 0
            edge_weight = edge_weight * deg_inv[col]
        elif normalization == 'row':
            row, _ = edge_index
            deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
            deg_inv = 1. / deg
            deg_inv[deg_inv == float('inf')] = 0
            edge_weight = edge_weight * deg_inv[row]
        elif normalization is None:
            pass
        else:
            raise ValueError(
                'Transition matrix normalization {} unknown.'.format(
                    normalization))

        return edge_index, edge_weight


    def diffusion_matrix_exact(self, edge_index, edge_weight, num_nodes,
                               method, **kwargs):
        r"""Calculate the (dense) diffusion on a given sparse graph.
        Note that these exact variants are not scalable. They densify the
        adjacency matrix and calculate either its inverse or its matrix
        exponential.

        Args:
            edge_index (LongTensor): The edge indices.
            edge_weight (Tensor): One-dimensional edge weights.
            num_nodes (int): Number of nodes.
            method (str): Diffusion method:

                1. :obj:`"ppr"`: Use personalized PageRank as diffusion.
                   Additionally expects the parameter:

                   - **alpha** (*float*) - Return probability in PPR.
                     Commonly lies in :obj:`[0.05, 0.2]`.

                2. :obj:`"heat"`: Use heat kernel diffusion.
                   Additionally expects the parameter:

                   - **t** (*float*) - Time of diffusion. Commonly lies in
                     :obj:`[2, 10]`.

                3. :obj:`"coeff"`: Freely choose diffusion coefficients.
                   Additionally expects the parameter:

                   - **coeffs** (*List[float]*) - List of coefficients
                     :obj:`theta_k` for each power of the transition matrix
                     (starting at :obj:`0`).

        :rtype: (:class:`Tensor`)
        """
        if method == 'ppr':
            # α (I_n + (α - 1) A)^-1
            edge_weight = (kwargs['alpha'] - 1) * edge_weight
            edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                     fill_value=1,
                                                     num_nodes=num_nodes)
            mat = to_dense_adj(edge_index, edge_attr=edge_weight).squeeze()
            diff_matrix = kwargs['alpha'] * torch.inverse(mat)

        elif method == 'heat':
            # exp(t (A - I_n))
            edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                     fill_value=-1,
                                                     num_nodes=num_nodes)
            edge_weight = kwargs['t'] * edge_weight
            mat = to_dense_adj(edge_index, edge_attr=edge_weight).squeeze()
            undirected = is_undirected(edge_index, edge_weight, num_nodes)
            diff_matrix = self.__expm__(mat, undirected)

        elif method == 'coeff':
            adj_matrix = to_dense_adj(edge_index,
                                      edge_attr=edge_weight).squeeze()
            mat = torch.eye(num_nodes, device=edge_index.device)

            diff_matrix = kwargs['coeffs'][0] * mat
            for coeff in kwargs['coeffs'][1:]:
                mat = mat @ adj_matrix
                diff_matrix += coeff * mat
        else:
            raise ValueError('Exact GDC diffusion {} unknown.'.format(method))

        return diff_matrix


    def diffusion_matrix_approx(self, edge_index, edge_weight, num_nodes,
                                normalization, method, **kwargs):
        r"""Calculate the approximate, sparse diffusion on a given sparse
        graph.

        Args:
            edge_index (LongTensor): The edge indices.
            edge_weight (Tensor): One-dimensional edge weights.
            num_nodes (int): Number of nodes.
            normalization (str): Transition matrix normalization scheme
                (:obj:`"sym"`, :obj:`"row"`, or :obj:`"col"`).
                See :func:`GDC.transition_matrix` for details.
            method (str): Diffusion method:

                1. :obj:`"ppr"`: Use personalized PageRank as diffusion.
                   Additionally expects the parameters:

                   - **alpha** (*float*) - Return probability in PPR.
                     Commonly lies in :obj:`[0.05, 0.2]`.

                   - **eps** (*float*) - Threshold for PPR calculation stopping
                     criterion (:obj:`edge_weight >= eps * out_degree`).
                     Recommended default: :obj:`1e-4`.

        :rtype: (:class:`LongTensor`, :class:`Tensor`)
        """
        if method == 'ppr':
            if normalization == 'sym':
                # Calculate original degrees.
                _, col = edge_index
                deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)

            edge_index_np = edge_index.cpu().numpy()
            # Assumes coalesced edge_index.
            _, indptr, out_degree = np.unique(edge_index_np[0],
                                              return_index=True,
                                              return_counts=True)
            indptr = np.append(indptr, len(edge_index_np[0]))

            neighbors, neighbor_weights = self.__calc_ppr__(
                indptr, edge_index_np[1], out_degree, kwargs['alpha'],
                kwargs['eps'])
            ppr_normalization = 'col' if normalization == 'col' else 'row'
            edge_index, edge_weight = self.__neighbors_to_graph__(
                neighbors, neighbor_weights, ppr_normalization,
                device=edge_index.device)
            edge_index = edge_index.to(torch.long)

            if normalization == 'sym':
                # We can change the normalization from row-normalized to
                # symmetric by multiplying the resulting matrix with D^{1/2}
                # from the left and D^{-1/2} from the right.
                # Since we use the original degrees for this it will be like
                # we had used symmetric normalization from the beginning
                # (except for errors due to approximation).
                row, col = edge_index
                deg_inv = deg.sqrt()
                deg_inv_sqrt = deg.pow(-0.5)
                deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
                edge_weight = deg_inv[row] * edge_weight * deg_inv_sqrt[col]
            elif normalization in ['col', 'row']:
                pass
            else:
                raise ValueError(
                    ('Transition matrix normalization {} not implemented for '
                     'non-exact GDC computation.').format(normalization))

        elif method == 'heat':
            raise NotImplementedError(
                ('Currently no fast heat kernel is implemented. You are '
                 'welcome to create one yourself, e.g., based on '
                 '"Kloster and Gleich: Heat kernel based community detection '
                 '(KDD 2014)."'))
        else:
            raise ValueError(
                'Approximate GDC diffusion {} unknown.'.format(method))

        return edge_index, edge_weight


    def sparsify_dense(self, matrix, method, **kwargs):
        r"""Sparsifies the given dense matrix.

        Args:
            matrix (Tensor): Matrix to sparsify.
            num_nodes (int): Number of nodes.
            method (str): Method of sparsification. Options:

                1. :obj:`"threshold"`: Remove all edges with weights smaller
                   than :obj:`eps`.
                   Additionally expects one of these parameters:

                   - **eps** (*float*) - Threshold to bound edges at.

                   - **avg_degree** (*int*) - If :obj:`eps` is not given,
                     it can optionally be calculated by calculating the
                     :obj:`eps` required to achieve a given :obj:`avg_degree`.

                2. :obj:`"topk"`: Keep edges with top :obj:`k` edge weights per
                   node (column).
                   Additionally expects the following parameters:

                   - **k** (*int*) - Specifies the number of edges to keep.

                   - **dim** (*int*) - The axis along which to take the top
                     :obj:`k`.

        :rtype: (:class:`LongTensor`, :class:`Tensor`)
        """
        assert matrix.shape[0] == matrix.shape[1]
        N = matrix.shape[1]

        if method == 'threshold':
            if 'eps' not in kwargs.keys():
                kwargs['eps'] = self.__calculate_eps__(matrix, N,
                                                       kwargs['avg_degree'])

            edge_index = (matrix >= kwargs['eps']).nonzero(as_tuple=False).t()
            edge_index_flat = edge_index[0] * N + edge_index[1]
            edge_weight = matrix.flatten()[edge_index_flat]

        elif method == 'topk':
            assert kwargs['dim'] in [0, 1]
            sort_idx = torch.argsort(matrix, dim=kwargs['dim'],
                                     descending=True)
            if kwargs['dim'] == 0:
                top_idx = sort_idx[:kwargs['k']]
                edge_weight = torch.gather(matrix, dim=kwargs['dim'],
                                           index=top_idx).flatten()

                row_idx = torch.arange(0, N, device=matrix.device).repeat(
                    kwargs['k'])
                edge_index = torch.stack([top_idx.flatten(), row_idx], dim=0)
            else:
                top_idx = sort_idx[:, :kwargs['k']]
                edge_weight = torch.gather(matrix, dim=kwargs['dim'],
                                           index=top_idx).flatten()

                col_idx = torch.arange(
                    0, N, device=matrix.device).repeat_interleave(kwargs['k'])
                edge_index = torch.stack([col_idx, top_idx.flatten()], dim=0)
        else:
            raise ValueError('GDC sparsification {} unknown.'.format(method))

        return edge_index, edge_weight


    def sparsify_sparse(self, edge_index, edge_weight, num_nodes, method,
                        **kwargs):
        r"""Sparsifies a given sparse graph further.

        Args:
            edge_index (LongTensor): The edge indices.
            edge_weight (Tensor): One-dimensional edge weights.
            num_nodes (int): Number of nodes.
            method (str): Method of sparsification:

                1. :obj:`"threshold"`: Remove all edges with weights smaller
                   than :obj:`eps`.
                   Additionally expects one of these parameters:

                   - **eps** (*float*) - Threshold to bound edges at.

                   - **avg_degree** (*int*) - If :obj:`eps` is not given,
                     it can optionally be calculated by calculating the
                     :obj:`eps` required to achieve a given :obj:`avg_degree`.

        :rtype: (:class:`LongTensor`, :class:`Tensor`)
        """
        if method == 'threshold':
            if 'eps' not in kwargs.keys():
                kwargs['eps'] = self.__calculate_eps__(edge_weight, num_nodes,
                                                       kwargs['avg_degree'])

            remaining_edge_idx = (edge_weight >= kwargs['eps']).nonzero(
                as_tuple=False).flatten()
            edge_index = edge_index[:, remaining_edge_idx]
            edge_weight = edge_weight[remaining_edge_idx]
        elif method == 'topk':
            raise NotImplementedError(
                'Sparse topk sparsification not implemented.')
        else:
            raise ValueError('GDC sparsification {} unknown.'.format(method))

        return edge_index, edge_weight


    def __expm__(self, matrix, symmetric):
        r"""Calculates matrix exponential.

        Args:
            matrix (Tensor): Matrix to take exponential of.
            symmetric (bool): Specifies whether the matrix is symmetric.

        :rtype: (:class:`Tensor`)
        """
        if symmetric:
            e, V = torch.symeig(matrix, eigenvectors=True)
            diff_mat = V @ torch.diag(e.exp()) @ V.t()
        else:
            diff_mat_np = expm(matrix.cpu().numpy())
            diff_mat = torch.Tensor(diff_mat_np).to(matrix.device)
        return diff_mat

    def __calculate_eps__(self, matrix, num_nodes, avg_degree):
        r"""Calculates threshold necessary to achieve a given average degree.

        Args:
            matrix (Tensor): Adjacency matrix or edge weights.
            num_nodes (int): Number of nodes.
            avg_degree (int): Target average degree.

        :rtype: (:class:`float`)
        """
        sorted_edges = torch.sort(matrix.flatten(), descending=True).values
        if avg_degree * num_nodes > len(sorted_edges):
            return -np.inf

        left = sorted_edges[avg_degree * num_nodes - 1]
        right = sorted_edges[avg_degree * num_nodes -1 ]
        return (left + right) / 2.0

    def __neighbors_to_graph__(self, neighbors, neighbor_weights,
                               normalization='row', device='cpu'):
        r"""Combine a list of neighbors and neighbor weights to create a sparse
        graph.

        Args:
            neighbors (List[List[int]]): List of neighbors for each node.
            neighbor_weights (List[List[float]]): List of weights for the
                neighbors of each node.
            normalization (str): Normalization of resulting matrix
                (options: :obj:`"row"`, :obj:`"col"`). (default: :obj:`"row"`)
            device (torch.device): Device to create output tensors on.
                (default: :obj:`"cpu"`)

        :rtype: (:class:`LongTensor`, :class:`Tensor`)
        """
        edge_weight = torch.Tensor(np.concatenate(neighbor_weights)).to(device)
        i = np.repeat(np.arange(len(neighbors)),
                      np.fromiter(map(len, neighbors), dtype=np.int))
        j = np.concatenate(neighbors)
        if normalization == 'col':
            edge_index = torch.Tensor(np.vstack([j, i])).to(device)
            N = len(neighbors)
            edge_index, edge_weight = coalesce(edge_index, edge_weight, N, N)
        elif normalization == 'row':
            edge_index = torch.Tensor(np.vstack([i, j])).to(device)
        else:
            raise ValueError(
                f"PPR matrix normalization {normalization} unknown.")
        return edge_index, edge_weight

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


def get_calc_ppr():
    import numba

    @numba.jit(nopython=True, parallel=True)
    def calc_ppr(indptr, indices, out_degree, alpha, eps):
        r"""Calculate the personalized PageRank vector for all nodes
        using a variant of the Andersen algorithm
        (see Andersen et al. :Local Graph Partitioning using PageRank Vectors.)

        Args:
            indptr (np.ndarray): Index pointer for the sparse matrix
                (CSR-format).
            indices (np.ndarray): Indices of the sparse matrix entries
                (CSR-format).
            out_degree (np.ndarray): Out-degree of each node.
            alpha (float): Alpha of the PageRank to calculate.
            eps (float): Threshold for PPR calculation stopping criterion
                (:obj:`edge_weight >= eps * out_degree`).

        :rtype: (:class:`List[List[int]]`, :class:`List[List[float]]`)
        """

        alpha_eps = alpha * eps
        js = [[0]] * len(out_degree)
        vals = [[0.]] * len(out_degree)
        for inode_uint in numba.prange(len(out_degree)):
            inode = numba.int64(inode_uint)
            p = {inode: 0.0}
            r = {}
            r[inode] = alpha
            q = [inode]
            while len(q) > 0:
                unode = q.pop()

                res = r[unode] if unode in r else 0
                if unode in p:
                    p[unode] += res
                else:
                    p[unode] = res
                r[unode] = 0
                for vnode in indices[indptr[unode]:indptr[unode + 1]]:
                    _val = (1 - alpha) * res / out_degree[unode]
                    if vnode in r:
                        r[vnode] += _val
                    else:
                        r[vnode] = _val

                    res_vnode = r[vnode] if vnode in r else 0
                    if res_vnode >= alpha_eps * out_degree[vnode]:
                        if vnode not in q:
                            q.append(vnode)
            js[inode] = list(p.keys())
            vals[inode] = list(p.values())
        return js, vals

    return calc_ppr


gdc = GDC(self_loop_weight=1, normalization_in='sym', normalization_out='col',
        diffusion_kwargs=dict(method='ppr', alpha=0.15),
        sparsification_kwargs={'avg_degree': 64, 'method': 'threshold'}, exact=True)


class TUDataset_aug(InMemoryDataset):
    r"""A variety of graph kernel benchmark datasets, *.e.g.* "IMDB-BINARY",
    "REDDIT-BINARY" or "PROTEINS", collected from the `TU Dortmund University
    <https://chrsmrrs.github.io/datasets>`_.
    In addition, this dataset wrapper provides `cleaned dataset versions
    <https://github.com/nd7141/graph_datasets>`_ as motivated by the
    `"Understanding Isomorphism Bias in Graph Data Sets"
    <https://arxiv.org/abs/1910.12091>`_ paper, containing only non-isomorphic
    graphs.

    .. note::
        Some datasets may not come with any node labels.
        You can then either make use of the argument :obj:`use_node_attr`
        to load additional continuous node attributes (if present) or provide
        synthetic node features using transforms such as
        like :class:`torch_geometric.transforms.Constant` or
        :class:`torch_geometric.transforms.OneHotDegree`.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The `name
            <https://chrsmrrs.github.io/datasets/docs/datasets/>`_ of the
            dataset.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        use_node_attr (bool, optional): If :obj:`True`, the dataset will
            contain additional continuous node attributes (if present).
            (default: :obj:`False`)
        use_edge_attr (bool, optional): If :obj:`True`, the dataset will
            contain additional continuous edge attributes (if present).
            (default: :obj:`False`)
        cleaned: (bool, optional): If :obj:`True`, the dataset will
            contain only non-isomorphic graphs. (default: :obj:`False`)
    """

    # url = ('http://ls11-www.cs.tu-dortmund.de/people/morris/'
    #        'graphkerneldatasets')
    url = 'https://www.chrsmrrs.com/graphkerneldatasets'
    cleaned_url = ('https://raw.githubusercontent.com/nd7141/'
                   'graph_datasets/master/datasets')

    def __init__(self, root, name, transform=None, pre_transform=None, pre_filter=None, 
                 use_node_attr=False, use_edge_attr=False,
                 cleaned=False, aug1=None, aug2=None, aug_ratio=0.5, num_parts1=None, num_parts2=None):
        self.name = name
        self.cleaned = cleaned
        super(TUDataset_aug, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        if self.data.x is not None and not use_node_attr:
            num_node_attributes = self.num_node_attributes
            self.data.x = self.data.x[:, num_node_attributes:]
        if self.data.edge_attr is not None and not use_edge_attr:
            num_edge_attributes = self.num_edge_attributes
            self.data.edge_attr = self.data.edge_attr[:, num_edge_attributes:]

        if not (self.name == 'MUTAG' or self.name == 'PTC_MR' or self.name == 'DD' or \
            self.name == 'PROTEINS' or self.name == 'NCI1' or self.name == 'NCI109' or \
            self.name == 'ENZYMES' or self.name == 'Mutagenicity'):
            edge_index = self.data.edge_index[0, :].numpy()
            _, num_edge = self.data.edge_index.size()
            nlist = [edge_index[n] + 1 for n in range(num_edge - 1) if edge_index[n] > edge_index[n + 1]]
            nlist.append(edge_index[-1] + 1)

            num_node = np.array(nlist).sum()
            self.data.x = torch.ones((num_node, 1))

            edge_slice = [0]
            k = 0
            for n in nlist:
                k = k + n
                edge_slice.append(k)
            self.slices['x'] = torch.tensor(edge_slice)

            '''
            print(self.data.x.size())
            print(self.slices['x'])
            print(self.slices['x'].size())
            assert False
            '''

        self.aug1 = aug1
        self.aug2 = aug2
        self.aug_ratio = aug_ratio
        self.num_parts1 = num_parts1
        self.num_parts2 = num_parts2

    @property
    def raw_dir(self):
        name = 'raw{}'.format('_cleaned' if self.cleaned else '')
        return osp.join(self.root, self.name, name)

    @property
    def processed_dir(self):
        name = 'processed{}'.format('_cleaned' if self.cleaned else '')
        return osp.join(self.root, self.name, name)

    @property
    def num_node_labels(self):
        if self.data.x is None:
            return 0
        for i in range(self.data.x.size(1)):
            x = self.data.x[:, i:]
            if ((x == 0) | (x == 1)).all() and (x.sum(dim=1) == 1).all():
                return self.data.x.size(1) - i
        return 0

    @property
    def num_node_attributes(self):
        if self.data.x is None:
            return 0
        return self.data.x.size(1) - self.num_node_labels

    @property
    def num_edge_labels(self):
        if self.data.edge_attr is None:
            return 0
        for i in range(self.data.edge_attr.size(1)):
            if self.data.edge_attr[:, i:].sum() == self.data.edge_attr.size(0):
                return self.data.edge_attr.size(1) - i
        return 0

    @property
    def num_edge_attributes(self):
        if self.data.edge_attr is None:
            return 0
        return self.data.edge_attr.size(1) - self.num_edge_labels

    @property
    def raw_file_names(self):
        names = ['A', 'graph_indicator']
        return ['{}_{}.txt'.format(self.name, name) for name in names]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        url = self.cleaned_url if self.cleaned else self.url
        folder = osp.join(self.root, self.name)
        path = download_url('{}/{}.zip'.format(url, self.name), folder)
        extract_zip(path, folder)
        os.unlink(path)
        shutil.rmtree(self.raw_dir)
        os.rename(osp.join(folder, self.name), self.raw_dir)

    def process(self):
        self.data, self.slices = read_tu_data(self.raw_dir, self.name)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))

    def get_num_feature(self):
        data = self.data.__class__()

        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[0]

        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key,
                                        item)] = slice(slices[0],
                                                       slices[0 + 1])
            else:
                s = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
        _, num_feature = data.x.size()

        return num_feature

    def get(self, idx):
        data = self.data.__class__()

        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[idx]

        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key,
                                        item)] = slice(slices[idx],
                                                       slices[idx + 1])
            else:
                s = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]

        node_num = data.edge_index.max()
        sl = torch.tensor([[n,n] for n in range(node_num)]).t()
        data.edge_index = torch.cat((data.edge_index, sl), dim=1)

        if self.aug1 == 'diffusion':
            data_aug1 = diffusion(deepcopy(data))
        elif self.aug1 == 'dnodes':
            data_aug1 = drop_nodes(deepcopy(data), self.aug_ratio)
        elif self.aug1 == 'pedges':
            data_aug1 = permute_edges(deepcopy(data), self.aug_ratio)
        elif self.aug1 == 'subgraph':
            data_aug1 = subgraph(deepcopy(data), self.aug_ratio)
        elif self.aug1 == 'mask_nodes':
            data_aug1 = mask_nodes(deepcopy(data), self.aug_ratio)
        elif self.aug1 == 'none':
            data_aug1 = deepcopy(data)
            data_aug1.x = torch.ones((data.edge_index.max()+1, 1))
        elif self.aug1 == 'random4':
            n = np.random.randint(3)
            if n == 0:
               data_aug1 = permute_edges(deepcopy(data), self.aug_ratio)
            elif n == 1:
               data_aug1 = mask_nodes(deepcopy(data), self.aug_ratio)
            elif n == 2:
               data_aug1 = drop_nodes(deepcopy(data), self.aug_ratio)
            elif n == 3:
               data_aug2 = diffusion(deepcopy(data))
            else:
                print('sample augmentation error')
                assert False
        else:
            print('augmentation error')
            assert False


        if self.aug2 == 'diffusion':
            data_aug2 = diffusion(deepcopy(data))
        elif self.aug2 == 'dnodes':
            data_aug2 = drop_nodes(deepcopy(data), self.aug_ratio)
        elif self.aug2 == 'pedges':
            data_aug2 = permute_edges(deepcopy(data), self.aug_ratio)
        elif self.aug2 == 'subgraph':
            data_aug2 = subgraph(deepcopy(data), self.aug_ratio)
        elif self.aug2 == 'mask_nodes':
            data_aug2 = mask_nodes(deepcopy(data), self.aug_ratio)
        elif self.aug2 == 'none':
            data_aug2 = deepcopy(data)
            data_aug2.x = torch.ones((data.edge_index.max()+1, 1))
        elif self.aug2 == 'random4':
            n = np.random.randint(4)
            if n == 0:
               data_aug2 = permute_edges(deepcopy(data), self.aug_ratio)
            elif n == 1:
               data_aug2 = mask_nodes(deepcopy(data), self.aug_ratio)
            elif n == 2:
               data_aug2 = drop_nodes(deepcopy(data), self.aug_ratio)
            elif n == 3:
               data_aug2 = diffusion(deepcopy(data))
            else:
                print('sample augmentation error')
                assert False
        else:
            print('augmentation error')
            assert False
        
        ## graph_cluster
        subgraph_a = cluster_data(data, num_parts=self.num_parts1, recursive=True)
        subgraph_b = cluster_data(data, num_parts=self.num_parts2, recursive=True)
        return data, data_aug1, data_aug2, subgraph_a, subgraph_b


def diffusion(data):
    data = gdc(data)
    return data

def drop_nodes(data, aug_ratio):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    drop_num = int(node_num * aug_ratio)

    idx_drop = np.random.choice(node_num, drop_num, replace=False)
    idx_nondrop = [n for n in range(node_num) if not n in idx_drop]
    idx_dict = {idx_nondrop[n]:n for n in list(range(node_num - drop_num))}

    # data.x = data.x[idx_nondrop]
    edge_index = data.edge_index.numpy()

    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    edge_index = adj.nonzero().t()

    data.edge_index = edge_index

    # edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)]
    # edge_index = [[edge_index[0, n], edge_index[1, n]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)] + [[n, n] for n in idx_nondrop]
    # data.edge_index = torch.tensor(edge_index).transpose_(0, 1)
    return data


def permute_edges(data, aug_ratio):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    permute_num = int(edge_num * aug_ratio)

    edge_index = data.edge_index.transpose(0, 1).numpy()

    idx_add = np.random.choice(node_num, (permute_num, 2))
    # idx_add = [[idx_add[0, n], idx_add[1, n]] for n in range(permute_num) if not (idx_add[0, n], idx_add[1, n]) in edge_index]

    # edge_index = np.concatenate((np.array([edge_index[n] for n in range(edge_num) if not n in np.random.choice(edge_num, permute_num, replace=False)]), idx_add), axis=0)
    # edge_index = np.concatenate((edge_index[np.random.choice(edge_num, edge_num-permute_num, replace=False)], idx_add), axis=0)
    edge_index = edge_index[np.random.choice(edge_num, edge_num-permute_num, replace=False)]
    # edge_index = [edge_index[n] for n in range(edge_num) if not n in np.random.choice(edge_num, permute_num, replace=False)] + idx_add
    data.edge_index = torch.tensor(edge_index).transpose_(0, 1)
    return data


def subgraph(data, aug_ratio):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    sub_num = int(node_num * aug_ratio)

    edge_index = data.edge_index.numpy()

    idx_sub = [np.random.randint(node_num, size=1)[0]]
    idx_neigh = set([n for n in edge_index[1][edge_index[0]==idx_sub[0]]])

    count = 0
    while len(idx_sub) <= sub_num:
        count = count + 1
        if count > node_num:
            break
        if len(idx_neigh) == 0:
            break
        sample_node = np.random.choice(list(idx_neigh))
        if sample_node in idx_sub:
            continue
        idx_sub.append(sample_node)
        idx_neigh.union(set([n for n in edge_index[1][edge_index[0]==idx_sub[-1]]]))

    idx_drop = [n for n in range(node_num) if not n in idx_sub]
    idx_nondrop = idx_sub
    idx_dict = {idx_nondrop[n]:n for n in list(range(len(idx_nondrop)))}

    # data.x = data.x[idx_nondrop]
    edge_index = data.edge_index.numpy()

    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    edge_index = adj.nonzero().t()

    data.edge_index = edge_index

    # edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)]
    # edge_index = [[edge_index[0, n], edge_index[1, n]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)] + [[n, n] for n in idx_nondrop]
    # data.edge_index = torch.tensor(edge_index).transpose_(0, 1)
    return data


def mask_nodes(data, aug_ratio):
    node_num, feat_dim = data.x.size()
    mask_num = int(node_num * aug_ratio)
    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    data.x[idx_mask] = torch.tensor(np.random.normal(loc=0.5, scale=0.5, size=(mask_num, feat_dim)), dtype=torch.float32)
    return data


def permute_data(data, perm, adj):
    data = copy.deepcopy(data)
    num_nodes = data.num_nodes

    for key, item in data:
        if item.size(0) == num_nodes:
            data[key] = item[perm]

    data.edge_index = None
    data.edge_attr = None
    data.adj = adj
    return data


def cluster_data(data, num_parts, recursive=True):
    ori_data = copy.deepcopy(data)

    (row, col), edge_attr = data.edge_index, data.edge_attr
    adj = SparseTensor(row=row, col=col, value=edge_attr)
    adj, partptr, perm = adj.partition(num_parts, recursive)
    
    # data = permute_data(data, perm, adj)

    sub_graph = []
    for idx in range(num_parts):
        start = int(partptr[idx])
        end = int(partptr[idx + 1])

        sub_data = copy.deepcopy(data)
        num_nodes = sub_data.num_nodes

        idx_sub = perm[start:end]
        idx_drop = [n for n in range(num_nodes) if not n in idx_sub]

        # for key, item in sub_data:
        #     if item.size(0) == num_nodes:
        #         sub_data[key] = item.narrow(0, start, length)

        edge_index = sub_data.edge_index.numpy()

        sub_adj = torch.zeros((num_nodes, num_nodes))
        sub_adj[edge_index[0], edge_index[1]] = 1
        sub_adj[idx_drop, :] = 0
        sub_adj[:, idx_drop] = 0
        edge_index = sub_adj.nonzero().t()

        sub_data.edge_index = edge_index

        if sub_data.edge_index.shape[1]==0:
            sub_data = ori_data

        sub_graph.append(sub_data)
    return sub_graph

