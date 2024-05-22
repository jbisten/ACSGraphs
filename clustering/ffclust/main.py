import argparse
import clustering
import IO
import IOFibers as IOF
import numpy as np
import logging
import ACSGraphs.clustering.utils.bundle2tck as b2t
import subprocess

def main():
    parser = argparse.ArgumentParser(description='Perform clustering on a dataset of streamlines')
    parser.add_argument('--points', nargs='*', type=int, default=list((0,3,10,17,20)),
                        help='Points to be used in map clustering')
    parser.add_argument('--ks', nargs='*', type=int, default=list((9, 5, 5, 5, 9)),
                        help='Number of clusters to be used for each point in K-Means for map')
    parser.add_argument('--thr-seg', type=float, default=120,
                        help='Minimum threshold for segmentation')
    parser.add_argument('--thr-join', type=float, default=120,
                        help='Minimum threshold for join')
    parser.add_argument('--outdir',
                        help='Directory where to place all output')
    parser.add_argument('--infile', help='Input streamlines file')
    args = parser.parse_args()

    bundles_dir = IO.create_output(args.outdir)

    # fibers = IO.read_bundles(args.infile)
    fibers = IO.read_bundles(args.infile)
    _, _, fiber_ids = IOF.read_bundles(args.infile)

    clusters, centroids, log, final_cluster_fiber_ids = clustering.fiber_clustering(fibers, fiber_ids, args.points,args.ks,args.thr_seg,args.thr_join)

    print(f'Number of cluster identified: {len(clusters)}')


    logging.basicConfig(level = logging.INFO, filename = args.outdir+"/stats.log")
    logging.info(log)

    IO.write_bundles(clusters, final_cluster_fiber_ids, centroids,bundles_dir,args.outdir)


if __name__=="__main__":
    main()
