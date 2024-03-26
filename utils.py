import os
import socket
import psutil
import re

import torch
import torch.distributed as dist

def init_comm_size_and_rank():
    world_size = None
    world_rank = 0

    if os.getenv("OMPI_COMM_WORLD_SIZE") and os.getenv("OMPI_COMM_WORLD_RANK"):
        ## Summit
        world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        world_rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
    elif os.getenv("SLURM_NPROCS") and os.getenv("SLURM_PROCID"):
        ## CADES
        world_size = int(os.environ["SLURM_NPROCS"])
        world_rank = int(os.environ["SLURM_PROCID"])

    ## Fall back to default
    if world_size is None:
        world_size = 1

    return int(world_size), int(world_rank)

def get_comm_size_and_rank():
    world_size = None
    world_rank = 0

    if dist.is_initialized():
        world_size = dist.get_world_size()
        world_rank = dist.get_rank()
    else:
        world_size = 1

    return int(world_size), int(world_rank)

def find_ifname(myaddr):
    """
    Find socket ifname for a given ip adress. This is for "GLOO" ddp setup.
    Usage example:
        find_ifname("127.0.0.1") will return a network interface name, such as "lo". "lo0", etc.
    """
    ipaddr = socket.gethostbyname(myaddr)
    ifname = None
    for nic, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.address == ipaddr:
                ifname = nic
                break
        if ifname is not None:
            break

    return ifname


def parse_slurm_nodelist(nodelist):
    """
    Parse SLURM_NODELIST env string to get list of nodes.
    Usage example:
        parse_slurm_nodelist(os.environ["SLURM_NODELIST"])
    Input examples:
        "or-condo-g04"
        "or-condo-g[05,07-08,13]"
        "or-condo-g[05,07-08,13],or-condo-h[01,12]"
    """
    nlist = list()
    for block, _ in re.findall(r"([\w-]+(\[[\d\-,]+\])*)", nodelist):
        m = re.match(r"^(?P<prefix>[\w\-]+)\[(?P<group>.*)\]", block)
        if m is None:
            ## single node
            nlist.append(block)
        else:
            ## multiple nodes
            g = m.groups()
            prefix = g[0]
            for sub in g[1].split(","):
                if "-" in sub:
                    start, end = re.match(r"(\d+)-(\d+)", sub).groups()
                    fmt = "%%0%dd" % (len(start))
                    for i in range(int(start), int(end) + 1):
                        node = prefix + fmt % i
                        nlist.append(node)
                else:
                    node = prefix + sub
                    nlist.append(node)

    return nlist


def setup_ddp():
    """ "Initialize DDP"""

    if os.getenv("DDP_BACKEND") is not None:
        backend = os.environ["DDP_BACKEND"]
    elif dist.is_nccl_available() and torch.cuda.is_available():
        backend = "nccl"
    elif torch.distributed.is_gloo_available():
        backend = "gloo"
    else:
        raise RuntimeError("No parallel backends available")

    world_size, world_rank = init_comm_size_and_rank()

    ## Default setting
    master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
    master_port = os.getenv("MASTER_PORT", "8889")

    if os.getenv("LSB_HOSTS") is not None:
        ## source: https://www.olcf.ornl.gov/wp-content/uploads/2019/12/Scaling-DL-on-Summit.pdf
        ## The following is Summit specific
        master_addr = os.environ["LSB_HOSTS"].split()[1]
    elif os.getenv("LSB_MCPU_HOSTS") is not None:
        master_addr = os.environ["LSB_MCPU_HOSTS"].split()[2]
    elif os.getenv("SLURM_NODELIST") is not None:
        ## The following is CADES specific
        master_addr = parse_slurm_nodelist(os.environ["SLURM_NODELIST"])[0]

    try:
        if backend in ["nccl", "gloo"]:
            os.environ["MASTER_ADDR"] = master_addr
            os.environ["MASTER_PORT"] = master_port
            os.environ["WORLD_SIZE"] = str(world_size)
            os.environ["RANK"] = str(world_rank)

        if (backend == "gloo") and ("GLOO_SOCKET_IFNAME" not in os.environ):
            ifname = find_ifname(master_addr)
            if ifname is not None:
                os.environ["GLOO_SOCKET_IFNAME"] = ifname

        print(
            "Distributed data parallel: %s master at %s:%s"
            % (backend, master_addr, master_port),
        )

        if not dist.is_initialized():
            dist.init_process_group(backend=backend, init_method="env://")

    except KeyError:
        print("DDP has to be initialized within a job - Running in sequential mode")

    return world_size, world_rank

def setup_ddp_groups(num_groups):

    world_size, world_rank = get_comm_size_and_rank()
    print ("world_size, world_rank:", world_size, world_rank)
    assert world_size >= num_groups, f'Number of groups {num_groups} requested exceeds number of total processes {world_size} available'

    num_ranks_per_group = world_size//num_groups

    ddp_groups_list = []
    for group_id in range(num_groups):
        ranks_list = [num_ranks_per_group*group_id + rank_id for rank_id in range(num_ranks_per_group)]
        ddp_groups_list.append(dist.new_group(ranks=ranks_list))

    for group_id in range(num_groups):
        if dist.get_rank(ddp_groups_list[group_id]) >= 0:
           print(f"Rank {world_rank} is in group {group_id}")

    return ddp_groups_list

def print0(*args, sep=" ", process_group=None):
    """
    Helper function to print/log messages.
    rank parameter is to limit which rank should print. if rank is None, all processes print.
    """
    rank = dist.get_rank(process_group)
    world_rank = dist.get_rank()

    if rank == 0:
        print(f"[{world_rank}:{rank}]", sep.join(map(str, args)))
