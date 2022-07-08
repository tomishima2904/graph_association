import torch
import sys
import numpy as np

class Comparators:

    def __init__(self, args) -> None:
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")
        pass


    def dot_product(self, all_tensors:torch.Tensor, query_tensors:torch.Tensor) -> np.ndarray:
        """
        Args:
            all_tensors (torch.Tensor): all vectors
            query_tensors (torch.Tensor): a query's vector
        """
        compared_tensors = torch.matmul(all_tensors, query_tensors)
        return compared_tensors.to('cpu').detach().numpy().copy()


    def __call__(self, all_vectors:dict, query_vectors:list) -> torch.Tensor:
        """
        Args:
            all_vectors (list): keys are all indicies, values are all indicies's vectors  [v, d]
            query_vectors (list): a query's vector  [d]
        Returns:
            np.ndarray: compared_tensors
        """

        all_tensors = torch.tensor(all_vectors, requires_grad=False)
        query_tensors = torch.tensor(query_vectors, requires_grad=False).to(self.device)

        if self.args.comparator == 'dot':
            return self.dot_product(all_tensors, query_tensors)
        else:
            print(f"Error: no such a comparator {self.args.comparator}")
            sys.exit(1)
