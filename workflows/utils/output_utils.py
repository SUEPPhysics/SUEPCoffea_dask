import os
import pathlib
import shutil
from typing import List, Optional
import h5py


def dump_table(
    fname: str, location: str, subdirs: Optional[List[str]] = None
) -> None:
    subdirs = subdirs or []
    xrd_prefix = "root://"
    pfx_len = len(xrd_prefix)
    xrootd = False
    if xrd_prefix in location:
        try:
            import XRootD
            import XRootD.client

            xrootd = True
        except ImportError:
            raise ImportError(
                "Install XRootD python bindings with: conda install -c conda-forge xrootd"
            )
    local_file = (
        os.path.abspath(os.path.join(".", fname))
        if xrootd
        else os.path.join(".", fname)
    )
    merged_subdirs = "/".join(subdirs) if xrootd else os.path.sep.join(subdirs)
    destination = (
        location + merged_subdirs + f"/{fname}"
        if xrootd
        else os.path.join(location, os.path.join(merged_subdirs, fname))
    )
    if xrootd:
        copyproc = XRootD.client.CopyProcess()
        copyproc.add_job(local_file, destination)
        copyproc.prepare()
        copyproc.run()
        client = XRootD.client.FileSystem(
            location[: location[pfx_len:].find("/") + pfx_len]
        )
        status = client.locate(
            destination[destination[pfx_len:].find("/") + pfx_len + 1 :],
            XRootD.client.flags.OpenFlags.READ,
        )
        assert status[0].ok
        del client
        del copyproc
    else:
        dirname = os.path.dirname(destination)
        if not os.path.exists(dirname):
            pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
        if not os.path.samefile(local_file, destination):
            shutil.copy2(local_file, destination)
        else:
            return
        assert os.path.isfile(destination)
    # delete the local file after copying it
    pathlib.Path(local_file).unlink()


def add_hists(file: str, hists: dict, group_name: str = 'hists') -> None:
    """
    Append histograms (hist.Hist) to an HDF5 file.
    Each histogram is stored in its own subgroup, with the histogram values, variances, and axis edges stored as datasets.
    :file str: path to the HDF5 file
    :hists dict: dictionary of histogram names and histogram objects to be added
    """

    with h5py.File(file, 'a') as hdf5_file:
        hist_collection = hdf5_file.create_group(group_name)

        for hist_name, hist in hists.items():
            # Create a subgroup for each histogram
            hist_group_path = f"{group_name}/{hist_name}"
            hist_group = hdf5_file.create_group(hist_group_path)
            
            # Store the histogram name as an attribute
            hist_group.attrs['name'] = hist_name
            
            # Store axis edges, values, and variances by converting them to numpy arrays
            for i, axis in enumerate(hist.axes):
                axis_name = f"axis{i}"
                hist_group.create_dataset(f"{axis_name}_name", data=axis.name.encode('utf-8'))
                hist_group.create_dataset(f"{axis_name}_edges", data=axis.edges)
            hist_group.create_dataset("values", data=hist.values())
            hist_group.create_dataset("variances", data=hist.variances())