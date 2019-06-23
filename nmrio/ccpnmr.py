import pandas as pd
import re
import os
import fnmatch



def ccp_tsv_in(f):
    """
    Description:  
    Given a filepath, f, to a *.tsv peak list saved through ccpnmr 
    imports the peak list and returns a pandas dataframe with columns by res 
    number and an index corresponding to features saved by ccpnmr and resnames.
    
    This function also handles unassigned residues with arbitrary numbers assigned
    by ccpnmr but returns a NaN for unnumbered residues.
    
    Inputs:
    f:  The filepath, a string.
    """
    
    # Read the raw file with pandas:
    df = pd.read_csv(f, sep='\t', usecols=[2,3,4,6,7,8,9], index_col=2)
    # Empty lists to store names and numbers:
    resn = list()
    index = list()
    # Pull out the name/number from the combo in the file:
    for i in df.index.values:
        name = re.findall(r'[A-Z][a-z]{2}', i)
        num = re.findall(r'\d+', i)
        # Dealing with unassigned residues:
        try:
            resn.append(name[0])
            index.append(num[0])
        except:
            # These are for non named residues:
            try:
                resn.append("X")
                index.append(num[0])
            except:
                # For residues with no name or number:
                index.append('0')
    # Unexpectedly got 'None' values for assignments in ccpnmr
    # Dealing with that here:
    resn = [x for x in resn if x != 'Non']
    # Adding in the residue name to the dataframe:
    df['resn'] = pd.Series(resn, index=df.index)
    # Replacing the original index with plain old numbers
    df.index = index
    df = df.drop_duplicates()
    df.index = df.index.astype(int)
    # Used to return a sorted axis but is less helpful with unassigned residues.
    return df.transpose()


def ccp_series_list_in(f):
    """
    An import function specific to a peak list generated by 
    the Data Series functionality of the ccp analysis software.
    
    Originally written to import V135G slow exchange peak intensity
    series.  May have wonky functionality with other stuff.
    """
    
    df = pd.read_csv(f, sep="\t", na_values='-')
    # Selecting columns corresponding to actual peak intensities
    df = df[df.columns[2:-2]]
    res, resn = list(), list()
    for i in df.index:
        res.append(int(re.findall(r'\d+', i)[0]))
        resn.append(re.findall(r'[A-Z]{1}[a-z]{2}', i)[0])
    df['resn'] = resn
    df.index = res
    return df.sort_index().transpose()



def read_tsv_peak_dir(data_dir, filter='*.tsv', slow=False):
    """
    A function to read all peak list files of the form '*list.tsv' in the 
    given data directory.  This version also has functionality hard coded
    in it that pulls the spectrometer frequency out, which is predicated
    on the filename starting with that frequency.
    """
   
    # Finding all peak list files, *list.tsv, in the current directory
    # Also pulling out the 
    df_list = list()
    keys = list()
    for f in os.listdir(data_dir):
        if fnmatch.fnmatch(f, filter):
            df_list.append(ccp_tsv_in(os.path.join(data_dir, f)))
            res = re.findall(r'\d+', f)[0]
            if slow == False:
                #keys.append(res)
                keys.append(f)
            elif slow == True:
                extra = re.findall(r'\_([A-Za-z])\_', f)[0]
                #keys.append(res + extra)
                keys.append(f)
    
    col_list = [set(df.columns.values) for df in df_list]
    cols = list(set.intersection(*col_list))
    cols = [i for i in cols if i != '0']
    df_list = [df[cols] for df in df_list]
    
    # Making a large dataframe from all peak list files
    big_df = pd.concat(df_list,axis=0, keys=keys, names=['peak list','feature'])
    return big_df

def shift_import(path, isdir=True):
    """
    Description:  A function that imports a peak list or lists and returns the 1H and 15N 
    chemical shift lists.
    
    Input:  path - path to the peak list or directory
    isdir; boolean True/False, true if a directory is passed, false if not.
    """
    
    if isdir == False:
        big_df = ccp_tsv_in(path)
        h1_shift = big_df.xs('Position F1')
        n15_shift = big_df.xs('Position F2')
    else:
        big_df = read_tsv_peak_dir(path)
        h1_shift = big_df.xs('Position F1', level='feature')
        n15_shift = big_df.xs('Position F2', level='feature')
    return h1_shift, n15_shift
