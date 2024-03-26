import torch
import torch.distributed as dist

from utils import *


def run(rank):
    tensor = torch.tensor([rank])

    # ## Not working if we do dist.new_group locally
    # subgroup_ranks = [ 0, 1, 2, 3 ] if rank < 4 else [ 4, 5, 6, 7 ]
    # subgroup = dist.new_group(ranks=subgroup_ranks)

    # if rank in subgroup_ranks:
    #     gather_list = [torch.tensor([0]) for _ in subgroup_ranks]
    #     dist.all_gather(gather_list, tensor, group=subgroup)
    #     print (rank, "gather_list:", gather_list)

    ## All members should know new group creation
    subgroup_ranks1 = [0, 1, 2, 3]
    subgroup_ranks2 = [4, 5, 6, 7]
    subgroup1 = dist.new_group(ranks=subgroup_ranks1)
    subgroup2 = dist.new_group(ranks=subgroup_ranks2)

    if rank in subgroup_ranks1:
        gather_list = [torch.tensor([0]) for _ in subgroup_ranks1]
        dist.all_gather(gather_list, tensor, group=subgroup1)
        print(rank, "gather_list:", gather_list)

    if rank in subgroup_ranks2:
        gather_list = [torch.tensor([0]) for _ in subgroup_ranks2]
        dist.all_gather(gather_list, tensor, group=subgroup2)
        print(rank, "gather_list:", gather_list)


if __name__ == "__main__":
    comm_size, rank = setup_ddp()
    print("DDP setup:", comm_size, rank)
    assert comm_size == 8, "This example is set to use 8 processes."

    run(rank)
    print("Done.")
