import pandas as pd, numpy as np
from rna_fold import RNAFold


'''
Adapted from notebook. Sorry it's messy. 

I wrote this to:
    1. take in a csv containing sequences with known structure
    2. use D-Wave SA to solve for each sequence
    3. compute specificity and sensitivity for D-Wave vs. Real results

If you want to skip step (2), it shouldn't be hard as long as you have
the list of stems from your algorithm.


'''

# Import pseudobase data and clean up lists
df = pd.read_csv('test_sets/pseudobase_data_clean.csv')
df['adj_stems'] = df['adj_stems'].apply(eval)

# Convert stems to base pairs
get_pairs = lambda x: [item for sublist in [[(i[0]+_,i[1]-_) for _ in range(i[2])] for i in x] for item in sublist]
df['pairs_cor'] = df['adj_stems'].apply(get_pairs)
df['total_BP'] = df['pairs_cor'].apply(len)

# Compute number of possible stems for each sequence
def compute_n_stems(seq):
    rna = RNAFold(seq,min_stem_len=4,min_loop_len=3,skip_params=True)
    return len(rna.stems)

df['alg_n_stems'] = df['seq'].apply(lambda x: compute_n_stems(str(x)))

# Subselect sequences with desired number of stems
# !! Might want to change this !!
df = df[(df['alg_n_stems']>=40) & (df['alg_n_stems']<100)]

# Remove sequences with short stems. This is helpful if you want to reduce 
# the size of the set
df = df[df['adj_stems'].apply(lambda x: min([_[2] for _ in x])>4)]


def compute_result(seq,c_B,c_L):
    '''
    Example compute function.

    '''
    rna = RNAFold(seq,min_stem_len=4,min_loop_len=3,skip_params=False,c_B=c_B,c_L=c_L)
    rna.compute_dwave_sa(sweeps=10000)
    
    return rna.best_score, rna.stems_used


def compute_n_stems(seq):
    rna = RNAFold(seq,min_stem_len=4,min_loop_len=3,skip_params=True)
    return len(rna.stems)


def correct_pairs(row,cB,cL):
    return len(set(row['pairs_alg_{}-{}'.format(cB,cL)]).intersection(set(row['pairs_cor'])))


def missing_pairs(row,cB,cL):
    return len(list(set(row['pairs_cor']) - set(row['pairs_alg_{}-{}'.format(cB,cL)])))


def extra_pairs(row,cB,cL):
    return len(list(set(row['pairs_alg_{}-{}'.format(cB,cL)]) - set(row['pairs_cor'])))


def sensitivity(row,cB,cL):
    return float(row['correct_BP_{}-{}'.format(cB,cL)]) / (  row['correct_BP_{}-{}'.format(cB,cL)] + row['missing_BP_{}-{}'.format(cB,cL)] )


def specificity(row,cB,cL):
    return float(row['correct_BP_{}-{}'.format(cB,cL)]) / (  row['correct_BP_{}-{}'.format(cB,cL)] + row['extra_BP_{}-{}'.format(cB,cL)] )


def calculations(df,cB,cL):
    df[ 'correct_BP_{}-{}'.format(cB,cL)] = df.apply(lambda x: correct_pairs(x,cB,cL), axis=1)  # TP
    df[ 'missing_BP_{}-{}'.format(cB,cL)] = df.apply(lambda x: missing_pairs(x,cB,cL), axis=1)  # FN
    df[   'extra_BP_{}-{}'.format(cB,cL)] = df.apply(lambda x: extra_pairs(x,cB,cL), axis=1)    # FP
    df['sensitivity_{}-{}'.format(cB,cL)] = df.apply(lambda x: sensitivity(x,cB,cL), axis=1)
    df['specificity_{}-{}'.format(cB,cL)] = df.apply(lambda x: specificity(x,cB,cL), axis=1)
    return df


for cB,cL in [[1,10]]:
    # Run D-Wave SA on all sequences and split the returned tuple into separate columns
    df['result_tup'] = df['seq'].apply(lambda x: compute_result(str(x),cB,cL))
    df['alg_score'] = df['result_tup'].apply(lambda x: x[0])
    df['alg_stems_{}-{}'.format(cB,cL)] = df['result_tup'].apply(lambda x: x[1])

    # Drop any that failed
    df.dropna(subset=['alg_stems_{}-{}'.format(cB,cL)],inplace=True)

    # Convert chosen stems (from algorithm) to base pairs
    df['pairs_alg_{}-{}'.format(cB,cL)] = df['alg_stems_{}-{}'.format(cB,cL)].apply(get_pairs)

    # Compute sensitivity and specificity
    df = calculations(df,cB,cL)

    print('''
    {b}, {l}

    specificity mean: {spm}
    sensitivity mean: {sem}

    '''.format(b=cB, l=cL, spm=df['specificity_{}-{}'.format(cB,cL)].mean(), sem=df['sensitivity_{}-{}'.format(cB,cL)].mean()))
