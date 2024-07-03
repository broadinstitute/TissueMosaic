from typing import Union, Optional, Iterator, Iterable, Generator
import numpy
from anndata import AnnData
from os import PathLike
import os.path
from pathlib import Path
import scipy
import pandas as pd
import gzip
import bz2


def anndata_from_expression_csv(filename: str, key: str, transpose: bool, top_n_rows: int = None):
    """
    Read a csv file with the expression data (i.e. count matrix) and returns an anndata object.
    To be used when your collaborators give you a .csv file instead of a .h5ad file.

    If :attr:`transpose == False`:
    The csv is expected to have a header: 'barcode', 'gene_name_1', ..., 'gene_name_N'.
    Each entry is expected to be something-like:  ACCDAT, 2, 0, ...., 1

    If :attr:`transpose == True`:
    The csv is expected to have a header: 'gene', 'barcode_name_1', ..., 'barcode_name_N'.
    Each entry is expected to be something-like: Arhgap18, 2, 0, ...., 1

    Args:
        filename: the path to the csv file to read
        key: the column name associated with the observations.
            It defaults to 'barcode' is :attr:`transpose` == False and 'gene' if :attr:`transpose` == True.
        transpose: bool, whether the matrix is gene_by_cell or cell_by_gene
        top_n_rows: int, the number of the top rows to read. Set to a small value (like 20) for debugging.

    Note:
        The output will always be cell_by_gene (i.e. cells=obs, genes=var) regardless the value of :attr:`transpose`

    Returns:
        adata: An anndata object with (i) anndata.X the counts in a scipy Compressed Sparse Row format
            (ii) anndata.obs the observation name (often the cellular barcodes)
            (iii) anndata.var the variable names (often the gene names)
    """

    def read_top_n_rows(_filename, _n_rows):
        _df_tmp = pd.read_csv(_filename, nrows=_n_rows)
        _dir_name = os.path.dirname(_filename)
        _basename = os.path.basename(_filename)
        _new_filename = os.path.join(_dir_name, "debug_"+_basename)
        _df_tmp.to_csv(_new_filename)
        return _new_filename

    if isinstance(top_n_rows, int) and 1 <= top_n_rows <= 1000:
        # this functionality is meant for debug. n_rows should be a small number, i.e. <= 1000
        new_filename = read_top_n_rows(filename, top_n_rows)
    else:
        # will read the entire file
        new_filename = filename

    # extract the name of the columns and check which column in the barcode column
    df_tmp = pd.read_csv(new_filename, nrows=2)
    col_names = list(df_tmp.columns)
    observation_col_index = col_names.index(key)

    # print("barcode_col_index -> {0}, len(col_names) -> {1}".format(barcode_col_index, len(col_names)))
    columns_to_read = numpy.arange(observation_col_index, len(col_names))
    first_column_names = observation_col_index is not None

    return read_text(new_filename,
                     transpose=transpose,
                     delimiter=',',
                     columns_to_read=columns_to_read,
                     first_column_names=first_column_names,
                     dtype='int16')
    

def read_text(
        filename: Union[PathLike, Iterator[str]],
        transpose: bool,
        delimiter: Optional[str] = None,
        columns_to_read: Optional[numpy.ndarray] = None,
        first_column_names: Optional[bool] = None,
        dtype: str = "float32") -> AnnData:
    """
    Read `.txt`, `.tab`, `.data` (text) file or csv files (in that case set delimiter=',')
    and returns an anndata object

    Args:
        filename: Data file, filename or stream.
        delimiter: Delimiter that separates data within text file. If `None`, will split at
            arbitrary number of white spaces, which is different from enforcing
            splitting at single white space `' '`.
        columns_to_read: An array with the integer index corresponding to the columns to read.
        first_column_names: Assume the first column stores row names (most likely the barcodes).
        dtype: Numpy data type for the data (not the column names).
        transpose: bool, where the data is cell_by_gene or gene_by_cell

    Returns:
        An anndata object.
    """
    if not isinstance(filename, (PathLike, str, bytes)):
        return _read_text(filename, delimiter, first_column_names, dtype, transpose)

    filename = Path(filename)
    if filename.suffix == ".gz":
        with gzip.open(str(filename), mode="rt") as f:
            return _read_text(f, delimiter, columns_to_read, first_column_names, dtype, transpose)
    elif filename.suffix == ".bz2":
        with bz2.open(str(filename), mode="rt") as f:
            return _read_text(f, delimiter, columns_to_read, first_column_names, dtype, transpose)
    else:
        with filename.open() as f:
            return _read_text(f, delimiter, columns_to_read, first_column_names, dtype, transpose)


def _iter_lines(file_like: Iterable[str]) -> Generator[str, None, None]:
    """ Helper for iterating only nonempty lines without line breaks """
    for line in file_like:
        line = line.rstrip("\r\n")
        if line:
            yield line
    
    
def _read_line_data(line, delimiter, columns_to_read):
    """ Helper to read only the columns you are interested in """
    line_numpy = numpy.array(line.split(delimiter))  
    if columns_to_read is None:
        line_data = line_numpy
    else:
        line_data = line_numpy[columns_to_read]
    return line_data
    

def _read_text(
        f: Iterator[str],
        delimiter: Optional[str],
        columns_to_read: Optional[numpy.ndarray],
        first_column_names: Optional[bool],
        dtype: str,
        transpose: bool) -> AnnData:

    # initialize the storage 
    comments = []
    data = []
    col_names = []
    row_names = []
    
    # iterator over the lines
    lines = _iter_lines(f)
    
    # read header and column names
    for line in lines:
        if line.startswith("#"):
            comment = line.lstrip("# ")
            if comment:
                comments.append(comment)
        else:
            if delimiter is not None and delimiter not in line:
                raise ValueError(f"Did not find delimiter {delimiter!r} in first line.")
            line_data = _read_line_data(line, delimiter, columns_to_read)
            
            # the first row could have the columns names in it
            if not is_float(line_data[-1]):
                col_names = line_data.tolist()
            else:
                if not is_float(line_data[0]) or first_column_names:
                    first_column_names = True
                    row_names.append(line_data[0])
                    data.append(line_data[1:].astype(dtype=dtype))
                else:
                    data.append(line_data.astype(dtype=dtype))
            break
           
    # try reading col_names from the last comment line
    if not col_names:
        if len(comments) > 0:
            col_names = numpy.array(comments[-1].split())
        else:
            # just numbers as col_names
            col_names = numpy.arange(len(data[0])).astype(str)
    col_names = numpy.array(col_names, dtype=str)
    
    # read another line to check if first column contains row names or not
    if first_column_names is None:
        first_column_names = False
        
    for line in lines:
        
        line_data = _read_line_data(line, delimiter, columns_to_read)     
        
        if first_column_names or not is_float(line_data[0]):
            first_column_names = True
            row_names.append(line_data[0])
            data.append(line_data[1:].astype(dtype=dtype))
        else:
            data.append(line_data.astype(dtype=dtype))
        break
    
    # if row names are just integers
    if len(data) > 1 and data[0].size != data[1].size:
        first_column_names = True
        col_names = numpy.array(data[0]).astype(int).astype(str)
        row_names.append(data[1][0].astype(int).astype(str))
        data = [data[1][1:]]
    
    # parse the file
    for line in lines:
        line_data = _read_line_data(line, delimiter, columns_to_read)     
        if first_column_names:
            row_names.append(line_data[0])
            data.append(line_data[1:].astype(dtype=dtype))
        else:
            data.append(line_data.astype(dtype=dtype))
    
    if data[0].size != data[-1].size:
        raise ValueError(
            f"Length of first line ({data[0].size}) is different "
            f"from length of last line ({data[-1].size})."
        )
    data = numpy.array(data, dtype=dtype)
    
    # transform row_names
    if not row_names:
        row_names = numpy.arange(len(data)).astype(str)
    else:
        row_names = numpy.array(row_names)
        for iname, name in enumerate(row_names):
            row_names[iname] = name.strip('"')
    
    # adapt col_names if necessary
    if col_names.size > data.shape[1]:
        col_names = col_names[1:]
    for iname, name in enumerate(col_names):
        col_names[iname] = name.strip('"')

    if transpose:
        return AnnData(
            scipy.sparse.csr_matrix(numpy.transpose(data)),
            obs=dict(obs_names=col_names),
            var=dict(var_names=row_names),
            dtype=dtype)
    else:
        return AnnData(
            scipy.sparse.csr_matrix(data),
            obs=dict(obs_names=row_names),
            var=dict(var_names=col_names),
            dtype=dtype)


def is_float(string):
    """
    Check whether string is float.
    See also
    --------
    http://stackoverflow.com/questions/736043/checking-if-a-string-can-be-converted-to-float-in-python
    """
    try:
        float(string)
        return True
    except ValueError:
        return False
