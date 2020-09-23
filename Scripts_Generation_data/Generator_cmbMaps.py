""" Code for creating CMB-maps from CLASS code,
    LensTools and Healpy

    Main author:  Hector Javier Hortua,
    orjuelajavier@gmail.com
    hortua.orjuela@rist.ro
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from classy import Class
import healpy as hp
from memory_profiler import profile
from math import pi, sin, cos, sqrt, log, floor, degrees
import numpy as np
from pympler import tracker
import pandas as pd
import json, csv
from sys import getsizeof
import multiprocessing
import os
import sys
import timeit
from astropy.units import deg
from lenstools import GaussianNoiseGenerator, CMBTemperatureMap
import astropy.units as u





def CLASS_output(cosmo, class_phys_params,l_max):
    """
    Function that generate  power spectrum using CLASS code

    Args:
        class_phys_params : (dict) see generate_CMB_map for its content
        l_max : maximum l

    Returns:
        (np.array,np.array) with the multipoles and the amplitudes $c_\l$
    """

    params_model = {
        'format': 'camb',
        'output': 'tCl lCl',
        'l_max_scalars': l_max,
        'lensing': 'yes',
        'YHe': 0.246
    }


    params_model.update(class_phys_params)

    cosmo.set(params_model)
    cosmo.compute()
    clss = cosmo.lensed_cl(l_max)
    ell = clss['ell']
    ps_tt = clss['tt']
    factor = ell * (ell + 1.) / (2. * pi)

    return ell, ps_tt


def rotate_map(hmap, rot_theta, rot_phi):
    """
    Take  healpix map array and return another healpix map array
    which is ordered such that it has been rotated in (theta, phi) by the
    amounts given.

    NB very slow, but verified.
    """

    nside = hp.npix2nside(len(hmap))

    t, p = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    # Define a rotator
    r = hp.Rotator(deg=True, rot=[rot_theta,rot_phi], coord='G')
    # Get theta, phi under rotated co-ordinates
    trot, prot = r(t, p)
    # Interpolate map onto these co-ordinates
    rot_map = hp.get_interp_val(hmap, trot, prot)

    return rot_map


def equidistributed(degree_side, tolerance):
    """
    Generate random point uniformly distributd on a sphere


    Args:
        degree_side: dimension of the patch in degrees
        tolerance:  space between two patches

    Returns:

    """

    N_count = 0
    a = 4 * np.pi ** 2 * (degree_side + tolerance) ** 2 / (360 ** 2)
    d = sqrt(a)
    M_v = int(round(np.pi / d))
    d_v = np.pi / M_v
    #print("estimated n, of points ",  M_v*M_psi )
    d_psi = a / d_v
    x, y, z, psi_angle, phi_angle = [], [], [], [], []
    for m in range(0, M_v - 1, 1):
        v = np.pi * (m + 0.5) / M_v
        M_psi = int(round(2 * np.pi * sin(v) / d_psi))
        for n in range(0, M_psi - 1, 1):
            psi = 2 * np.pi * n / M_psi
            N_count += 1
            phi_angle.append(degrees(v))
            psi_angle.append(degrees(psi))

    return psi_angle, phi_angle, N_count

def generate_CMB_map(
        phys_parameters_dict, l_max, Nside=2048,
        sigma_noise= None,healpy_map=False):
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

        l_max: maximum l
        Nside: size of the pixels you have in a sphere
        healpy_map: Use healpy

    Returns:
        helapix map of the CMB if healpy_map is True,
        otherwise return the power spectrum and l's

    """

    CMB_standard_fit = {
        'h':       0.67556,
        #'100*theta_s':1.042143,
        'omega_b': 0.022032,
        'omega_cdm': 0.12038,
        'A_s':       2.215e-9,
        'n_s':        0.9619,
        'tau_reio':    0.0925
                        }

    phys_parameters_dict_completed = CMB_standard_fit.copy()
    phys_parameters_dict_completed.update(phys_parameters_dict)
    ell_1, ps_tt_1 =[],[]

    cosmo = Class()
    ell, ps_tt = CLASS_output(cosmo, phys_parameters_dict_completed, l_max)
    ell_1, ps_tt_1 = ell, ps_tt
    temperature_sq = (2.7255e6) ** 2

    def arcmin2rad(x):
        return x / 60 / 360 * 2 * np.pi
    if sigma_noise is None:
        sigma_noise_r= None
    else:
        sigma_noise_r= arcmin2rad(sigma_noise)

    if healpy_map:
         mask1 = hp.synfast(temperature_sq * ps_tt, Nside,lmax=l_max,
               mmax=None, alm=False, pol=False, pixwin=True, fwhm=0.0,
               sigma=sigma_noise_r, new=False, verbose=False)
    del  ell
    del  ps_tt
    cosmo.struct_cleanup()

    cosmo.empty()
    import gc
    gc.collect()

    if healpy_map:
        return mask1
    else:
        return ell_1, temperature_sq * ps_tt_1


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
        'h': [],
        #'100*theta_s':[1.0,1.3],
        'omega_b':[0.019,0.031],
        'omega_cdm':[0.06,0.2],
        'A_s': [1.015e-9,  4.015e-9],
        'n_s': [], # TODO
        'tau_reio':[]

                        }
    CMB_standard_fit = {
        'h':       0.67556,
        #'100*theta_s':1.042143,
        'omega_b': 0.022032,
        'omega_cdm': 0.12038,
        'A_s':       2.215e-9,
        'n_s':        0.9619,
        'tau_reio':    0.0925
                        }
    parameter_list = ['h', 'omega_b', 'omega_cdm', 'A_s', 'n_s', 'tau_reio']


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
        self._patch_size = data_generator_parameters['patch_size']
        self._pic_size = data_generator_parameters['pic_size']
        self._augm_data = data_generator_parameters['augm_data']
        self._pic_per_run_rot = data_generator_parameters['pic_per_run_rot']
        self._pic_per_run_equator = data_generator_parameters['pic_per_run_equator']
        self._pic_per_run_latitude = data_generator_parameters['pic_per_run_latitude']
        self._sigma_noise = data_generator_parameters['Gaussian_sigma_noise']
        self._healpy_used = data_generator_parameters['Use_Healpy']
        self._parameters_per_pic = None
        self.id = self.generator_id(data_generator_parameters)
        self.create_directories_and_files()
        self.time_start = timeit.default_timer()

    def set_parameters_per_pic(self):
        """ create the dataframe of phsical parameters"""

        self._parameters_per_pic = sample_parameters(
                                self._params,  self.parameter_ranges,
                                self._n_samples)

    def maps_generator(self,  l_max=2500, Nside=2048, nproc=38):
        """
        The map generator take information from the manager and
        generate all the images.  and terminate everyrthing


        Args:
            l_max:max number of multipolar numbers
            Nside:nmber of pixel you have in a sphere
            nproc: number of processes used for the rotations

        """


        if not self._parameters_per_pic:
            self.set_parameters_per_pic()

        for ix, row in self._parameters_per_pic.iterrows():
            parameters_dict = row.to_dict()
            # we can run the same picture several time
            for run in range(self._n_runs):
                #self.time_start = timeit.default_timer()
                if self._healpy_used == True:
                    cmb_map = generate_CMB_map(parameters_dict, l_max,
                            Nside, self._sigma_noise,
                            healpy_map=self._healpy_used)
                    print('map generated in ',timeit.default_timer()
                            - self.time_start )

                    if self._pic_per_run_rot != 0:
                        if self._pic_per_run_rot == -1:
                            # we take all the possible rotations
                            #  tolerance is kept fixed.
                            ang_Psi_arr, ang_Phi_arr, N_count = \
                                equidistributed(
                                    self._patch_size, 0.1)
                        elif self._pic_per_run_rot == 1:
                            ang_Psi_arr = [np.random.uniform(0, np.pi)]
                            ang_Phi_arr = [np.random.uniform(0, np.pi*2)]
                            N_count = 1
                        else:
                            raise ValueError('The value must be -1, 0 or 1')
                        self.rotations_cmb_img_gen(cmb_map, ang_Psi_arr,
                                                ang_Phi_arr,
                                                parameters_dict,run,
                                                self._pic_per_run_equator,
                                                self._pic_per_run_latitude,
                                                 nproc)
                    else:
                        self.extract_patch(cmb_map, parameters_dict, run, 0,
                                           self._pic_per_run_equator,
                                           self._pic_per_run_latitude)
                    import gc
                    del cmb_map
                    plt.close('all')
                    gc.collect()
                    print('execution run {} ended in {} seconds'.format(
                    run,timeit.default_timer() -self.time_start))


                else:
                    ell_power, power_map = generate_CMB_map(
                                        parameters_dict, l_max, Nside,
                                        self._sigma_noise,
                                        healpy_map=self._healpy_used)
                    self.extract_patch_Lens_Tools(ell_power,power_map,
                                        parameters_dict, run)
        self.label_file.close()

    def rotations_cmb_img_gen(
                self, full_map, ang_Psi_arra, ang_Phi_arra,
                parameters, run, number_patch_equator,
                number_patch_latitude, nproc=1):
        """
        for a list of angles it extract all the patches

        Args:
            full_map:
            ang_Psi_arra: list of latitudinal angles
            ang_Phi_arra: list of longitudinal angles
            parameters: (dict) of te modified parameters
            run:
            number_patch_equator:
            number_patch_latitude:
            nproc (int): number of processes to use to parallelize
                            rotations (default 1, single process)

        Returns:

        """

        def rotate_and_extract(full_map, psi, phi, parameters, run,
                    ix_mrotation, number_patch_equator,
                     number_patch_latitude):
            print("I am {} EXTRACT!".format(ix_mrotation))
            s2 = rotate_map(full_map, psi, phi)
            self.extract_patch(s2, parameters, run, ix_mrotation,
            number_patch_equator, number_patch_latitude)

            return 0

        angles = zip(ang_Psi_arra, ang_Phi_arra)
        pool = multiprocessing.Pool()
        print("before pooling")
        for (ix, (psi, phi)) in enumerate(angles):
            rotate_and_extract(full_map, psi, phi, parameters,
            run, ix, number_patch_equator, number_patch_latitude)
        pool.close()
        pool.join()

    def extract_patch(self, s2, parameters, run,
            ix_mrotation, number_patch_long, number_patch_lati):
          """
          Extract patches from s2 picture and save them

          Args:
              s2: healpy whole CMB map
              parameters: cosmological paramters
              run: run of the simulation
              ix_mrotation: an index for the rotation
              number_patch_long: how many patches on the horizontal axis to be
                         extracted (2l+1, the distance of the neightbohrood)
              number_patch_lati: how many patches on the vertical axis to be
                    extracted (2l+1 rows, the distance of the neightbohrood)


          """
          patch_size_deg = self._patch_size
          Longitude_initial = patch_size_deg * (number_patch_long - 1)
          Latitude_initial = patch_size_deg * (number_patch_lati - 1)
          position = 0
          for longitude_1 in np.arange(-Longitude_initial -(patch_size_deg*0.5),
                                   Longitude_initial -((patch_size_deg)*0.5)+1,
                                   patch_size_deg):
              for latitude_1 in np.arange(-Latitude_initial-(patch_size_deg*0.5),
                                     Latitude_initial -((patch_size_deg)*0.5)+1,
                                      patch_size_deg):
                  latitude_2 = latitude_1 + patch_size_deg
                  longitude_2 = longitude_1 + patch_size_deg
                  mean_lati = (latitude_1 + latitude_2) *0.5
                  mean_long = (longitude_1 + longitude_2) *0.5
                  position += 1
                  is_center = False
                  error = patch_size_deg / 1000
                  if (-error < mean_lati <error) and (-error< mean_long <error):
                      is_center = True
                  if self._augm_data:
                      n_rotations = 3
                      angle_rota = 90 * n_rotations
                      for rot in range(0, angle_rota + 1, 90):
                          pic = self.create_picture(s2, mean_lati, mean_long,
                                                     latitude_2,
                                                     longitude_2,
                                                     latitude_1, longitude_1,
                                                     rot)
                          pic = pic.data
                          pic = np.flip(pic, 0)
                          self.data_saver(pic, parameters, run, ix_mrotation,
                                          is_center, position,
                                          {'rot_angle': rot, 'flip':False})
                          # and flip:
                          self.data_saver(np.fliplr(pic), parameters, run,
                                          ix_mrotation,is_center, position,
                                          {'rot_angle': rot, 'flip':True})
                  else:
                      rot = 0
                      pic = self.create_picture(s2, mean_lati, mean_long,
                                                latitude_2, longitude_2,
                                          latitude_1, longitude_1, rot)
                      pic = pic.data
                      pic = np.flip(pic, 0)
                      self.data_saver(pic, parameters, run, ix_mrotation,
                                    is_center, position,
                                    {'rot_angle': 0, 'flip':False})


    def create_picture(self, complete_map ,mean_latia, mean_longa,
            latitude_2a, longitude_2a, latitude_1a, longitude_1a, rota):
        """ Wrapper of the LensTools function. It returns the picture"""

        return hp.visufunc.cartview(map=complete_map, fig=None, coord=['G','C'],
                                     rot=[mean_longa, mean_latia, rota],
                                     zat=None, unit='',
                                     lonra=[longitude_1a, longitude_2a],
                                     latra=  [latitude_1a, latitude_2a],
                                     nest=False, remove_dip=False,
                                     remove_mono=False, gal_cut=0, min=None,
                                     max=None, flip='astro', format='%.3g',
                                     cbar=False, cmap=None, norm=None,
                                     aspect=None,  hold=False, sub=None,
                                     margins=None, title='',
                                     notext=True, xsize=self._pic_size,
                                     return_projected_map=True)

    def extract_patch_Lens_Tools(self, ell_power, power_map, parameters, run):
          """
          Extract patches from s2 picture and save them

          Args:
              ell_power:array of multipolar numbers
              parameters: cosmological paramters
              run: run of the simulation
              ix_mrotation: only for hp

          """

          patch_size_deg = self._patch_size
          picture_size = self._pic_size
          noise_lever =self._sigma_noise
          is_center = True
          if self._augm_data:
              for rot in range(0, 4):
                  pic = self.create_picture_Lens_Tools(ell_power, power_map,
                             patch_size_deg, picture_size, noise_lever)
                  pic = pic.data
                  pic =np.rot90(pic,rot)
                  self.data_saver(pic, parameters, run,0,
                                  is_center, 0,
                                  {'rot_angle': rot, 'flip':False})
                  self.data_saver(np.fliplr(pic),parameters, run,0, is_center,0,
                                  {'rot_angle': rot, 'flip':True})

          else:
              pic = self.create_picture_Lens_Tools(ell_power, power_map,
                        patch_size_deg, picture_size, noise_lever)
              pic = pic.data
              self.data_saver(pic, parameters, run, 0, is_center,
                              0, {'rot_angle': 0, 'flip':False})


    def create_picture_Lens_Tools(self, ell_power, power_map,
               patch_size_deg, picture_size, noise_lever):
        """ Wrapper of the LensTools function. It returns the picture"""

        side_angle = patch_size_deg*deg
        gen = GaussianNoiseGenerator(
            shape=(picture_size,picture_size), side_angle=side_angle)
        gaussian_map = gen.fromConvPower(
                    np.array([ell_power,power_map]),seed=None,kind="linear",
                    bounds_error=False,fill_value=0.0)
        gaussian_map.smooth(
        noise_lever*u.arcmin, kind="gaussianFFT",inplace=True)

        return gaussian_map.data


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
        use_hp = data_generator_parameters['Use_Healpy']
        if use_hp is True:
            file_string_dir = "_hp"
        else:
            file_string_dir = "_lt"
        for i in not_used_parameters:
            all_parameters_list.remove(i)
        # create the id
        id = "_".join(all_parameters_list)
        id += ("-" + data_generator_parameters['par_distr'])
        id += (
            "-pppr_rot" +str( data_generator_parameters['pic_per_run_rot']))
        id +=  (
            "-pppr_eq" + str(data_generator_parameters['pic_per_run_equator']))
        id += (
            "-pppr_lat" +str(data_generator_parameters['pic_per_run_latitude']))
        id += ("-aug" + str(data_generator_parameters['augm_data']))
        id += ("-cod" + str(file_string_dir))
        id += ("+patch_s" +str( data_generator_parameters['patch_size']))
        id += ("+pic_s" +str( data_generator_parameters['pic_size']))

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
                        ('params', 'par_distr', 'pic_per_run_rot',
                         'pic_per_run_equator', 'pic_per_run_latitude',
                         'augm_data', 'patch_size','pic_size')}

        info_file = self.dir_4_save + 'params.json'
        with open(info_file, 'w') as file:
            info_to_save['params'] = list(info_to_save['params'])
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
            picture: numpy array of the patch
            parameters: physical parameters to be written
            run: (int) specific index for the run
            ix_mrotation: (int) specific index fir the spherical rotation
            is_center: (bool) a flag for the central picture
            position: (int) position with respect to the equator
            augm_params: (dict) if the picture is in a augmented  dataset
            by rotations and flips
                'rot_angle': degree of rotation (none is 0)
                'flip': (bool) if it is flipped

        """

        all_parameters_to_write = self.CMB_standard_fit.copy()
        all_parameters_to_write.update(parameters)

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
        if self._healpy_used:
            file_string += "_hp"
        else:
            file_string += "_lt"
        file_string += ".npy"
        all_parameters_to_write['filename'] = file_string
        #ipdb.set_trace()
        np.save(self.dir_4_save+file_string, picture)
        self.label_csv_writer.writerow(all_parameters_to_write)
        self.label_file.flush()


if __name__ == "__main__":

        l_max = 2500

        Nside = 2048
        # number of partitions on sphere

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
             "params": { 'h' ,'omega_b' ,'omega_cdm','A_s','n_s','tau_reio'}
            'par_distr' (str) lattice  or uniform describes the distribution of
             the parameter in the range
            'Gaussian_sigma_noise':(float) smooth images. In units of arcmin
            'pic_per_run_rot' ([-1(all path)], [0(no rotatecmbmap)] ,
            [1(random_angle)] )
             number of pict per run taken rotating the full map
             (all none or a random one)
            'pic_per_run_equator'(int) number of pictures per run at the equator
            'patch_size' : patch_size in degrees. If larger than 20 is better
                            Use_Healpy==False
            'pic_size' : (int) pixel per picture (side)
            'debug_mode' (bool) load only the first 500 pictures
            'Use_Healpy':(bool) if False, the maps are created using LensTools,
                        otherwise Healpy will make the task
        """
        data_generator_parameters = {
            "data_dir": '/ssd_data/CMB/',
            "n_runs" : 60,
            "n_parameter_samples": 1000,
            "params": {'omega_b' ,'omega_cdm','A_s'},
            'Gaussian_sigma_noise':0.0,
            "par_distr": 'uniform',
            'pic_per_run_rot':0 ,
            'pic_per_run_equator': 1,
            'pic_per_run_latitude': 1,
            "augm_data": 0,
            'patch_size': 10,
            'pic_size': 256,
            'parallel': False,
            'Use_Healpy' : False
        }
        """
         one gets (n_parameter_samples*n_runs) images if pic_per_run_rot=0
         and gets (n_parameter_samples*n_runs) images*8 if augm_data=1

        """
        mg = MapGenerator(data_generator_parameters)
        mg.maps_generator(nproc=nproc)
