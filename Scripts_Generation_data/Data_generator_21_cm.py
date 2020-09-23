""" Code for creating 21cm brightness_temp 
    Main author:  Hector Javier Hortua,
    hortua.orjuela@rist.ro
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging, sys

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger('21cmFAST')
logger.setLevel(logging.INFO)
import py21cmfast as p21c
from py21cmfast import plotting
from py21cmfast import cache_tools
from math import pi, sin, cos, sqrt, log, floor, degrees
import numpy as np
import pandas as pd
import json, csv
from sys import getsizeof
import multiprocessing
import os
import sys
import timeit
from astropy.units import deg
import astropy.units as u
import shutil






def clear_data(path):
    for root, dirs, files in os.walk(path):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))


def generate_CMB_map(
        phys_parameters_dict, initial_z, final_z,
        samples_z, HII_DIM,BOX_LEN,DIM,path_cache_21cm):
    """
    generate the CMB_map. It uses the standard LambdaCDM parameters if not
    passed in the dictionary phys_parameters_dict

    Args:
        phys_parameters_dict:
            'h' : Hubble parameter (for normalization purpose)
            'omega_b' : omega baryonic matter
            'omega_cdm' :omega cold dark matter
            'A_s': amplitude of the primordial spectrum
            'n_s': spectral index
            'tau_reio': opacity at the recombination


    Returns:
        coeval map of 21cm

    """
    path_cache=path_cache_21cm+'cache/'
    if not os.path.isdir(path_cache):
        os.makedirs(path_cache)

    sigma_8 = phys_parameters_dict['sigma_8']
    omega_m = phys_parameters_dict['omega_m']
    HII_EFF_FACTOR = phys_parameters_dict['HII_EFF_FACTOR']
    ION_Tvir_MIN = phys_parameters_dict['ION_Tvir_MIN']


    redshifts_values = [i for i in np.linspace(initial_z,final_z,samples_z)]
    # generate the curve from physical parameters
    coeval= p21c.run_coeval(
    redshift = redshifts_values,
    user_params = {"HII_DIM": HII_DIM, "BOX_LEN": BOX_LEN,'DIM':DIM},
    cosmo_params = p21c.CosmoParams(SIGMA_8=sigma_8,OMm=omega_m),
    astro_params = p21c.AstroParams({"HII_EFF_FACTOR": HII_EFF_FACTOR,"ION_Tvir_MIN":ION_Tvir_MIN}),
    cleanup=True,
    write=True,
    regenerate=True,
    direc=path_cache,
    random_seed=None)

    stacks=[0]*len(redshifts_values)
    for i in range(len(redshifts_values)):
        stacks[i] = np.take(coeval[i].brightness_temp, 0, axis=-1).T
    image_21cm = np.stack((stacks), axis=-1)
    clear_data(path_cache)

    return image_21cm


def sample_parameters(parameters2sample, parameter_ranges,n_samples):
    """
    uniformly sample the parameters within the given range

    Args:
        parameters2sample: class parameters to be sampled
        parameter_ranges:  ramge of the class parameters
        n_samples:  number of samples

    Returns:
        pandas dataframe with sampled parameters

    """
    n_parameters = len(parameters2sample)
    array_shape = (n_samples, n_parameters)

    mins = []
    maxs = []

    for par in parameters2sample:
        mins.append(parameter_ranges[par][0])
        maxs.append(parameter_ranges[par][1])
    par_array = np.random.uniform(mins,maxs,array_shape)
    sampled_parameters = pd.DataFrame(par_array,columns=parameters2sample)

    return sampled_parameters



class MapGenerator:
    """ A class for managing the systematic generation of the dataset"""

    parameter_ranges = {
        'ION_Tvir_MIN':[4.6,5.6],
        'HII_EFF_FACTOR':[10,100],
        'sigma_8':[0.6,0.8],
        'omega_m':[0.2,0.4] # TODO
                        }

    parameter_list = ['ION_Tvir_MIN', 'HII_EFF_FACTOR', 'sigma_8', 'omega_m']


    def __init__(self,data_generator_parameters):
        """

        Args:
            data_generator_parameters: (dict) dictionatry with all the
            parameters needed for the generation of images.
        """

        self._data_dir = data_generator_parameters['data_dir']
        self.dir_4_save = None
        self.file_prepared = False
        self.label_file_name = None
        self.label_file = None
        self.label_csv_writer = None
        self._data_generator_parameters = data_generator_parameters
        self._parallel = data_generator_parameters['parallel']
        self._n_runs = data_generator_parameters['n_runs']
        self._params = data_generator_parameters['params']
        self._n_samples = data_generator_parameters['n_parameter_samples']
        self._initial_z = data_generator_parameters['initial_z']
        self._final_z = data_generator_parameters['final_z']
        self._samples_z = data_generator_parameters['samples_z']
        self._HII_DIM = data_generator_parameters['HII_DIM']
        self._BOX_LEN = data_generator_parameters['BOX_LEN']
        self._DIM = data_generator_parameters['DIM']
        self.id = self.generator_id(data_generator_parameters)
        self._augm_data = data_generator_parameters['augm_data']
        self.create_directories_and_files()
        self.time_start = timeit.default_timer()
        self._parameters_per_pic = None

    def set_parameters_per_pic(self):
        """ create the dataframe of phsical parameters"""

        self._parameters_per_pic = sample_parameters(
                                self._params,  self.parameter_ranges,
                                self._n_samples)

    def maps_generator(self,  nproc=38):
        """
        The map generator take information from the manager and
        generate all the images.  and terminate everyrthing


        Args:

            nproc: number of processes used for the rotations

        """

        # generage the points
        if not self._parameters_per_pic:
            self.set_parameters_per_pic()
        # for every sampled parameters point
        for ix, row in self._parameters_per_pic.iterrows():
            parameters_dict = row.to_dict()
            for run in range(self._n_runs):

                power_map  = generate_CMB_map(parameters_dict,
                                    self._initial_z, self._final_z,
                                    self._samples_z, self._HII_DIM,
                                    self._BOX_LEN,self._DIM,
                                    self._data_dir)
                self.extract_patch_Lens_Tools(power_map,
                                        parameters_dict, run)
        self.label_file.close()


    def extract_patch_Lens_Tools(self,  power_map, parameters, run):
          """
          Extract patches from s2 picture and save them

          Args:


              run: run of the simulation


          """

          is_center = True
          if self._augm_data:
              for rot in range(0, 4):
                  pic =np.rot90(power_map,rot)
                  self.data_saver(pic, parameters, run,0,
                                  is_center, 0,
                                  {'rot_angle': rot, 'flip':False})
                  self.data_saver(np.fliplr(pic),parameters, run,0, is_center,0,
                                  {'rot_angle': rot, 'flip':True})

          else:
              self.data_saver(power_map, parameters, run, 0, is_center,
                              0, {'rot_angle': 0, 'flip':False})



    @classmethod
    def generator_id(cls,data_generator_parameters):
        """
        create an id for the generator process

        Args:
            data_generator_parameters: (dict) parameters

        Returns:
            (str) the id

        """

        all_parameters_list = cls.parameter_list.copy()


        parameters_sampled = data_generator_parameters['params']
        all_parameters_set = set(cls.parameter_list)

        not_used_parameters = all_parameters_set.difference(parameters_sampled)

        for i in not_used_parameters:
            all_parameters_list.remove(i)
        # create the id
        id = "_".join(all_parameters_list)
        id += ("-" + data_generator_parameters['par_distr'])
        id += (
            "-iz" +str( data_generator_parameters['initial_z']))
        id +=  (
            "-fz" + str(data_generator_parameters['final_z']))
        id +=  (
            "-samz" + str(data_generator_parameters['samples_z']))
        id += (
            "-pxl" +str(data_generator_parameters['HII_DIM']))
        id += ("-sMpc" + str(data_generator_parameters['BOX_LEN']))


        return id


    def create_directories_and_files(self):
        """ create the directories from the id and prepare the label files"""

        if not os.path.isdir(self._data_dir):
            os.makedirs(self._data_dir)
        if not os.path.isdir(self._data_dir+self.id):
            os.makedirs(self._data_dir+self.id)
        else:
            print('WARNING: The generator was already done')
        self.dir_4_save = self._data_dir+self.id+'/'
        # save the info of this generator
        info_to_save = {k: data_generator_parameters[k] for k in
                        ('params', 'par_distr', 'initial_z',
                         'final_z', 'samples_z',
                         'augm_data', 'HII_DIM','BOX_LEN')}
        info_to_save['pic_size']=None

        info_file = self.dir_4_save + 'params.json'
        with open(info_file, 'w') as file:
            info_to_save['params'] = list(info_to_save['params'])
            info_to_save['pic_size'] = info_to_save['HII_DIM']
            json.dump(info_to_save, file)

        # prepare the csv writer
        parameter_file_name = 'labels_file.csv'
        self.label_file_name = self.dir_4_save + parameter_file_name

        self.label_file = open(self.label_file_name, 'w', newline='')


        csv_field = self.parameter_list + ['run','ix_mrotation', 'is_center',
                    'position', 'rot_angle','flip', 'filename' ]

        self.label_csv_writer = csv.DictWriter(
            self.label_file, delimiter='\t',
            quotechar='|', quoting=csv.QUOTE_MINIMAL,fieldnames= csv_field)
        self.label_csv_writer.writeheader()
        print("Dirs  have been created... ")
        self.file_prepared = True



    def data_saver(
            self, picture, parameters, run, ix_mrotation,
            is_center, position, augm_params):
        """
        A saver that save pictures and label and gives te appropriate name

        Args:

            parameters: physical parameters to be written
            run: (int) specific index for the run
            ix_mrotation: (int) specific index fir the spherical rotation
            augm_params: (dict) if the picture is in a augmented  dataset
            by rotations and flips
                'rot_angle': degree of rotation (none is 0)
                'flip': (bool) if it is flipped

        """

        all_parameters_to_write=parameters

        all_parameters_to_write['run'] = run
        all_parameters_to_write['ix_mrotation'] = ix_mrotation
        all_parameters_to_write['is_center'] = is_center
        all_parameters_to_write['position'] = position

        all_parameters_to_write['rot_angle'] = augm_params['rot_angle']
        all_parameters_to_write['flip'] = augm_params['flip']

        # construction of the file name
        parameter_string_lst = [
                             str(all_parameters_to_write[k])
                             for k in self.parameter_list]
        file_string = "_".join(parameter_string_lst)
        file_string += "-r" + str(all_parameters_to_write['run'])
        file_string += "-mr" + str(all_parameters_to_write['ix_mrotation'])
        file_string += "_p" + str(all_parameters_to_write['position'])
        file_string += "_ra" + str(all_parameters_to_write['rot_angle'])
        file_string += "_f" + str(all_parameters_to_write['flip'])

        if is_center:
            file_string += "_CNT"
        if all_parameters_to_write['rot_angle'] == 0 and (not all_parameters_to_write['flip'] ):
            file_string += "_ORIG"
        file_string += ".npy"
        all_parameters_to_write['filename'] = file_string
        #ipdb.set_trace()
        np.save(self.dir_4_save+file_string, picture)
        self.label_csv_writer.writerow(all_parameters_to_write)
        self.label_file.flush()


if __name__ == "__main__":


        if sys.argv[1] == 'pool':
            nproc = 38
        elif sys.argv[1] == "single":
            nproc = 1
        else:
            raise ValueError("not an option!")

        """
        'data_dir' (str)    path (relative or absolute) to the dataset
            'augm_data' (bool or int) if true use also reflected and rotated
             pictures
            'params_dim' (int) dim of parameter space in the datasets
            'par_distr' (str) lattice  or uniform describes the distribution of
             the parameter in the range
            'pic_size' : (int) pixel per picture (side)
            'debug_mode' (bool) load only the first 500 pictures

        """
        data_generator_parameters = {
            "data_dir": '/ssd_data/CMB/Dataset21_cm-ot/',
            "n_runs" : 7,
            "n_parameter_samples": 600,
             "params": { 'sigma_8' ,'omega_m' ,'HII_EFF_FACTOR','ION_Tvir_MIN'},
            "par_distr": 'uniform',
            "augm_data": 0,
            'initial_z':6,
            'final_z':12 ,
            'samples_z':20,
            'HII_DIM': 128,
            'BOX_LEN': 192,
            'DIM': 384,
            'parallel': False,
            'dir_cache_21':'/ssd_data/CMB/Dataset21_cm/c0'

        }
        """
         one gets (n_parameter_samples*n_runs) images if pic_per_run_rot=0
         and gets (n_parameter_samples*n_runs) images*8 if augm_data=1

        """
        mg = MapGenerator(data_generator_parameters)
        mg.maps_generator(nproc=nproc)
