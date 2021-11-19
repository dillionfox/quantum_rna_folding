#!/usr/bin/env python

import numpy as np
from random import uniform
import itertools, time, sys
from rna_fold import RNAFold


'''
Compute all possible solutions to an Ising model.

I ran this serially and got the following scaling, measured in seconds:

y(n_cores) = 4.08e-06 * exp(0.7 * n_cores)

Examples:
28 qubits: 00:23:38
30 qubits: 01:36:20
32 qubits: 06:32:31


MPI Scaling (ballpark based on best performing cores):
=====================================================
time ~ 2**N_qubits/(#cores*50000) seconds

Example Run:
===========
mpiexec -np 16 python3 exact_solver.py MODEL ARG1 ARG2 ARG3
mpiexec -np 16 python3 exact_solver.py MockModel N_qubits
mpiexec -np 16 python3 exact_solver.py RNAModel seq <min_stem_len> <min_loop_len>

'''

class Model(object):

    """ Model Base Class """

    def __init__(self):

        h: dict = None
        J: dict = None

        ## Number of qubits required for model
        self.N_qubits: int = None 

        ## NP 2D matrix from h/J
        model: numpy.ndarray = None

        ## Perform class level consistency checks
        self.check()

    def check(self):
        pass

    def execute(self):
       
        ## Compute H & J 
        self._compute_h_and_J()
        self._dicts_to_np()

    def _compute_h_and_J(self):

        """ Virtual method for computing h & J 
        
            This method should be replaced with the ACTUAL method
            for computing h & J 
        
        """

        self.h = None
        self.J = None
        self.N_qubits = 0

        return self.h, self.J, self.N_qubits

    def _dicts_to_np(self):
        
        """ Converts h/J dicts to 2D np array"""

        self.model = np.zeros((self.N_qubits,self.N_qubits), dtype='float32')
        for i in range(self.N_qubits):
            for j in range(i,self.N_qubits):
                if i==j:
                    self.model[i][i] = self.h[i]
                else:
                    self.model[i][j] = self.J[(i,j)]
        return self.model

    def score(self, num):
        """ Compute score for passed bit vector """

        ## Convert integer to bit representation 
        b = np.unpackbits(np.array([num], dtype='>i8').view(np.uint8))[-self.N_qubits:]
        #b = np.array(list(np.binary_repr(num).zfill(self.N_qubits))).astype(np.int8) 
        b = b.reshape(-1, b.shape[0])
        result = np.einsum('ij,ik,jk->i',b,b,self.model)
        return b[result.argmin()], result.min()

    def result(self):
        """ Return a result """
        return None

class RNAModel(Model):

    """ Derived model class to compute RNA model """

    def __init__(self, seq: str, min_stem_len: int = 3, min_loop_len: int = 3):
        super().__init__()

        ## Instantiate RNA class
        self.rna_ss = RNAFold(seq, min_stem_len = int(min_stem_len), 
                min_loop_len = int(min_loop_len))

        print(f'N Stems: {len(self.rna_ss.stems)}')
        sys.stdout.flush()

    def _compute_h_and_J(self):
        self.h = self.rna_ss.h
        self.J = self.rna_ss.J
        self.N_qubits = len(self.rna_ss.stems) 
        return self.h, self.J, self.N_qubits

    def result(self, bv):
        """ Return the result """
        return np.array(self.rna_ss.stems)[bv.astype('bool')] 

class MockModel(Model):

    def __init__(self, N_qubits: int):
        super().__init__()
        
        self.N_qubits = int(N_qubits)

    def _compute_h_and_J(self):

        self.h = {i : uniform(-1,1) for i in range(self.N_qubits)}
        self.J = {(i,j) : uniform(-1,1) for i in range(self.N_qubits) for j in range(i+1,self.N_qubits)}
        return self.h, self.J, self.N_qubits

    def result(self, bv):
        """ Return the result """
        return None

class ExactSolver(object):
 
    def __init__(self, model_name: str, model_args: list):

        ## The model used for scoring
        self.model: Model     = None
        self.model_name: str  = model_name 
        self.model_args: list = model_args 

        ## Timer
        self.t0 = None 
        
        ## MPI
        self.mpi_enabled = False
        self.comm = None
        self.size: int = None
        self.rank: int = None

        ## Go
        self.check()

    def check(self):

        ## Check to see if we've got an MPI job
        self._init_mpi()

    def execute(self):

        ## Check if serial or parallel
        if self.size > 1:
            result = self._run_mpi()
        else:
            result = self._run_serial()

    def _init_mpi(self):

        """ Checks to see if we're MPI enabled """

        try:
            from mpi4py import MPI
        except Exception as e:
            self.comm = None
            self.rank = 0
            self.size = 1
            print('Unable to initialize MPI: {e}') 
            print(f'MPI Status: Rank: {self.rank} Size: {self.size}') 
            return
        else:
            self.comm = MPI.COMM_WORLD
            self.size = self.comm.Get_size() 
            self.rank = self.comm.Get_rank()
            print(f'MPI Status: Rank: {self.rank} Size: {self.size}') 
            sys.stdout.flush()

    def _grouper_it(self):
        it = iter(self._bits)
        while True:
            chunk_it = itertools.islice(it, self.rank, None, self.size)
            try:
                first_el = next(chunk_it)
            except StopIteration:
                return
            yield [_ for _ in itertools.chain((first_el,), chunk_it)]

    def _build_bit_vectors(self):
        self._bits = itertools.product([0,1], repeat=self.model.N_qubits)

    def _timestamp(self):

        if self.t0 == None:
           self.t0 = time.time() 

        return time.time()-self.t0

    def _run_serial(self):
        """ Run solver serially """

        self._timestamp() 
 
        ## Initialize the model 
        self.model = globals()[self.model_name](*self.model_args)
        self.model.execute()
       
        ## Array of integers 
        gen=itertools.count(self.rank, self.size)
        
        stop = 2**self.model.N_qubits
        best_score = ([0], 1e100)
        steps=0
        for num in gen:
            if num < stop:
                result, score = self.model.score(num)
                if score < best_score[1]:
                   best_score = (result, score)
            else:
                break

        print(f'{self.model_name},{best_score},{self.size},{self._timestamp()}') 

    def _run_mpi(self):
        """ Run solver with MPI """

        if self.rank == 0:

            self._timestamp() 
            
            ## Initialize the model 
            self.model = globals()[self.model_name](*self.model_args)
            self.model.execute()

        else:
            self.model = None

        ## Broadcast model/bits to other procs        
        self.model = self.comm.bcast(self.model, root=0)

        ## Work on only the slice relevent to this rank
        gen=itertools.count(self.rank, self.size)
        stop = 2**self.model.N_qubits
        steps=0
        best_score = ([0], 1e100)
        for num in gen:
            if num < stop:
                result, score = self.model.score(num)
                if score < best_score[1]:
                   best_score = (result, score)
            else:
                break

            if steps % 250000 == 0:
                print(f'Rank: {self.rank}, step: {steps}, progress: {100 * self.size * steps/stop:.2f} %,'
                        f' time: {self._timestamp():.4f}, steps_per_second: {steps / self._timestamp():.0f},'
                        f' time remaining: {(((stop/self.size)-steps)*self._timestamp()/(steps+1)):.4f}, best: {best_score}')
                sys.stdout.flush()

            steps += 1

        ## Gather from all processes and compute final best score.
        print(f'DONE Rank: {self.rank}, step: {steps}, progress: {100 * self.size * steps/stop:.2f} %,'
                f' time: {self._timestamp():.4f}, steps_per_second: {steps / self._timestamp():.0f},'
                f' time remaining: {(((stop/self.size)-steps)*self._timestamp()/(steps+1)):.4f}, best: {best_score}')
        sys.stdout.flush()
        self.comm.barrier() 
        best_score_all = self.comm.gather(best_score, root=0)
        if self.rank == 0:
            best_score_all = min(best_score_all, key=lambda x: x[1])
            print(f'{self.model_name},{best_score_all},{self.model.result(best_score_all[0])},{self.size},{self._timestamp()}') 
            sys.stdout.flush()

if __name__ == "__main__":

    model_name = sys.argv[1]
    model_args = sys.argv[2:] 

    solver = ExactSolver(model_name, model_args)
    solver.execute()
