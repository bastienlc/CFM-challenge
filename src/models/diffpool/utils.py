import torch


def get_segment_indices(batch: torch.Tensor):
    """
    Given a batch, returns a list of tuples (a, b) where a is the index of the first element of the segment and b is the index of the first element of the next segment.
    """
    x = batch[0]
    start = 0

    output = []
    for end, y in enumerate(batch[1:]):
        if y != x:
            output.append((start, end + 1))
            x = y
            start = end
    output.append((start, len(batch)))
    return output


def batch_diffpool(S: torch.Tensor, Z: torch.Tensor, A: torch.Tensor):
    """
    Computes diffpool operations for batches of S, Z and A. Using this function supposes that all graphs in the batch have the same number of nodes. This is the case for all diffpool layers except the first one.

    Parameters:
    -----------
    S : torch.Tensor
        Batch of S matrices of shape (batch_size x n_l x n_{l+1})
    Z : torch.Tensor
        Batch of Z matrices of shape (batch_size x n_l x d)
    A : torch.Tensor
        Batch of A matrices of shape (batch_size x n_l x n_l)

    Returns:
    --------
    X_out : torch.Tensor
        Batch of X matrices of shape (batch_size * n_{l+1} x d)
    A_out : torch.Tensor
        Batch of A matrices of shape (batch_size * n_{l+1} x batch_size * n_{l+1})
    """

    batch_size, size = S.shape[0], S.shape[2]

    X_out = torch.bmm(S.transpose(1, 2), Z)
    A_out = torch.bmm(torch.bmm(S.transpose(1, 2), A), S)

    return X_out.reshape(batch_size * size, -1), torch.block_diag(*A_out)


def extract_blocks(A: torch.Tensor, size: int, batch_size: int):
    """
    A: (batch_size * size x batch_size * size)
    Returns: (batch_size x size x size)
    """

    # Extract the batch_size diagonal blocks of shape size x size
    # without for loop
    return A.reshape(batch_size, size, batch_size, size)[
        torch.arange(batch_size), :, torch.arange(batch_size), :
    ]
