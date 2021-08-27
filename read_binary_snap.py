# *************************************************************************************************
#                                Modulo de Programas Lectores 
# *************************************************************************************************
import os
import gzip
import numpy as np


# -----------------------------------------------------------------------------------------------
#                                         Default Quantities
# -----------------------------------------------------------------------------------------------
flags = {   
    'flag_rho'             : True,
    'flag_sfr'             : True,
    'flag_hsml'            : True,
    'flag_metals'          : True,
    'flag_energy'          : True,
    'flag_cooling'         : True,
    'flag_feedback'        : True,
    'flag_EnergySN'        : True,
    'flag_stellarage'      : True,
    'flag_EnergySNCold'    : True,
    'flag_artificial'      : True,
    'flag_speciesKrome'    : True,
    'flag_sn'              : False,
    'flag_bhx'             : False,
    'flag_outputpotential' : False,
}


# -----------------------------------------------------------------------------------------------
#                                         Functions
# -----------------------------------------------------------------------------------------------

def uncompress(fname):    ## Uncompress file .gz

    with gzip.GzipFile(fname) as f: s = f.read()
    fname = fname[:-3]
    with open(fname,'wb') as f: f.write(s)
    return fname


def read_header():
   
    dummy = np.fromfile(f,'i',1)[0]
    header = {}

    header['npart']    = np.fromfile(f,'i',6)    # Number of particles [Ngas, Nhalo, Ndisk, Nbulge, Nstars, Ntracers]  
    header['massarr']  = np.fromfile(f,'d',6)    # Mass indicator: 0 if mass varies, 1 if mass fixed
    header['time']     = np.fromfile(f,'d',1)[0] # time [internal unity: 9.8e8 yr/h]
    header['redshift'] = np.fromfile(f,'d',1)[0]

    header['flag_sfr']      = np.fromfile(f,'i',1)[0]
    header['flag_feedback'] = np.fromfile(f,'i',1)[0]

    header['npartTotal'] = np.fromfile(f,'I',6)  # equal to npart, but sum the gas masses that did not become stars

    header['flag_cooling'] = np.fromfile(f,'i',1)[0] 
    header['num_files']    = np.fromfile(f,'i',1)[0]
    
    header['BoxSize']     = np.fromfile(f,'d',1)[0]
    header['Omega0']      = np.fromfile(f,'d',1)[0] 
    header['OmegaLambda'] = np.fromfile(f,'d',1)[0]
    header['HubbleParam'] = np.fromfile(f,'d',1)[0]
    
    header['flag_stellarage'] = np.fromfile(f,'i',1)[0]
    header['flag_metals']     = np.fromfile(f,'i',1)[0]
    
    header['npartTotalHighWord'] = np.fromfile(f,'I',6)
    
    header['flag_entropy_instead_u'] = np.fromfile(f,'i',1)[0]
    header['flag_doubleprecision']   = np.fromfile(f,'i',1)[0]
    header['flag_ic_info'] = np.fromfile(f,'i',1)[0]
    
    header['lpt_scalingfactor'] = np.fromfile(f,'f',1)[0]
    header['fill']  = np.fromfile(f,'b',18)
    header['names'] = np.fromfile(f,'b',30)

    dummy = np.fromfile(f,'i',1)[0]     
    
    return header


def read_quantity(data_type, n_data):

    dummy = np.fromfile(f,'i',1)[0]
    quantity = np.fromfile(f, data_type, n_data)
    dummy = np.fromfile(f,'i',1)[0]
    
    return quantity


def read_info_block(n_block, display_n_block=False):
    
    dummy = np.fromfile(f,'i',1)[0]
    info = np.full(n_block, {})
    for i in range(n_block):
        info[i] = {
        'label'     : ''.join([chr(item) for item in np.fromfile(f,np.int8,4)]),
        'type'      : ''.join([chr(item) for item in np.fromfile(f,np.int8,8)]),
        'ndim'      : np.fromfile(f,'i',1)[0],
        'is_present': np.fromfile(f,'i',6)
        }    
    dummy = np.fromfile(f,'i',1)[0]
    if display_n_block: print('Readed info; n_block={:d}'.format(n_block))
    
    return info


################ Read file for "merger_test43_new.py" ################
def read(fname, flags=flags, header_only=False, IC_only=False, control=False):
    """
    Read a snapshot in a binary file 
    
    Parameters
    ----------
    fname: string
           Path for snap to read
           
    flags: dict, optional
           Flag for quantities to read from snapshot
           Default:
           flags = {   
                    'flag_rho'             : True,
                    'flag_sfr'             : True,
                    'flag_hsml'            : True,
                    'flag_metals'          : True,
                    'flag_energy'          : True,
                    'flag_cooling'         : True,
                    'flag_feedback'        : True,
                    'flag_EnergySN'        : True,
                    'flag_stellarage'      : True,
                    'flag_EnergySNCold'    : True,
                    'flag_artificial'      : True,
                    'flag_speciesKrome'    : True,
                    'flag_sn'              : False,
                    'flag_bhx'             : False,
                    'flag_outputpotential' : False,
                   }
     
     header_only: bool, optional
           Return just the header. Default: False 
           
     IC_only: bool, optional
           Return just the necesary quantities for an IC. Default: False (CAUTION: testing)
     
     control: bool, optional
           Print control of quantities. Default: False
    
    Returns
    -------
    header: dict
           Snapshot header
           
    data: dict, optional: if "only_header"=False
           Snapshot quantity data
           
    info: dict, optional: if "only_header"=False and there is quantities information in file
           Snapshot quantity information    
    """
    
    
    compress_file = '.gz' in fname   
    if compress_file: fname = uncompress(fname)  # Uncompress file
    
    try: 
        global f
        with open(fname,'r') as f:               # open and browse the file

            header = read_header()
            npart = header['npart']
            massarr = header['massarr']
            ntypart = len(npart)
            
            Ngas, Nhalo, Ndisk, Nbulge, Nstars, Nbound = npart ; N = int(np.sum(npart))

            if header_only: return header        # Return just the header
            
            data = {}
            for i,j in enumerate(npart): 
                if j>0: data['PartType{:01d}'.format(i)] = {}
            
            n_block = 0
            ind = [i for i in range(ntypart) if npart[i] > 0 and massarr[i] == 0] 
            Nwithmass = np.sum(npart[ind])

            # Particle positions [internal unity: 3.085678e21 cm = 1.0 kpc/h]
            pos = read_quantity('f',3*N).reshape(N,3) ; n_block += 1

            # Particle velocities [internal unity: 1e5 cm/seg = 1.0 km/seg]
            vel = read_quantity('f',3*N).reshape(N,3) ; n_block += 1               

            # Particle ID numbers [internal unity: 1e5 cm/seg = 1.0 km/seg]
            id1 = read_quantity('I',N) ; n_block += 1

            # Mass for particles with variable mass [internal unity: 1.989e43 g = 1e10 M_sol/h]
            if Nwithmass > 0: mass = read_quantity('f',Nwithmass)
            else: mass=np.zeros(1)
            n_block += 1

            if IC_only: return(fname, N, Ngas, Nhalo, Ndisk, Nbulge, Nstars, time, redshift, id1, pos, vel, mass)
            
            if Ngas > 0:
                
                # Specific internal energy of SPH particles [internal unity: (km/seg)^2]
                if flags['flag_energy']: data['PartType0']['InternalEnergy'] = read_quantity('f',Ngas) ; n_block += 1  

                # Comovil density of SPH particles
                if flags['flag_rho']: data['PartType0']['Density'] = read_quantity('f',Ngas) ; n_block += 1   

                if flags['flag_cooling']: 
                    data['PartType0']['ne1'] = read_quantity('f',Ngas) ; n_block += 1
                    data['PartType0']['nh']  = read_quantity('f',Ngas) ; n_block += 1

                # Smoothing length of SPH particles
                if flags['flag_hsml']: data['PartType0']['SmoothingLength'] = read_quantity('f',Ngas) ; n_block += 1    

                if flags['flag_sfr']: data['PartType0']['StarFormationRate'] = read_quantity('f',Ngas) ; n_block += 1

                if Nstars > 0: 
                    # Time of birth of New Stars
                    if flags['flag_stellarage']: 
                        data['PartType4']['StellarFormationTime'] = read_quantity('f',Nstars) ; n_block += 1

                    if flags['flag_bhx']: data['PartType4']['NBlackHoles'] = read_quantity('f',Nstars) ; n_block += 1

                if flags['flag_metals']: 
                    Zm = read_quantity('f',12*(Ngas+Nstars)).reshape(Ngas+Nstars,12) ; n_block += 1
                    data['PartType0']['ElementAbundance'] = Zm[:Ngas]
                    data['PartType4']['ElementAbundance'] = Zm[Ngas:]                    
                    
                if flags['flag_speciesKrome']:
                    data['PartType0']['speciesKrome'] = read_quantity('f',16*Ngas).reshape(Ngas,16) ; n_block += 1
                    data['PartType0']['gammaKrome']   = read_quantity('f',Ngas) ; n_block += 1
                    data['PartType0']['muKrome']      = read_quantity('f',Ngas) ; n_block += 1

                if flags['flag_outputpotential']: Pot = read_quantity('f',N) ; n_block += 1
                else: Pot = []
                
                if flags['flag_artificial']:
                    data['PartType0']['ArtViscosityCoef']    = read_quantity('f',Ngas) ; n_block += 1 # artficial viscosity
                    data['PartType0']['ArtConductivityCoef'] = read_quantity('f',Ngas) ; n_block += 1 # artficial conductivity
                
                if flags['flag_EnergySN']: 
                    EnergySN = read_quantity('f',Ngas+Nstars) ; n_block += 1
                    data['PartType0']['EnergySN'] = EnergySN[:Ngas]
                    data['PartType4']['EnergySN'] = EnergySN[Ngas:]
                if flags['flag_EnergySNCold']: 
                    EnergySNCold = read_quantity('f',Ngas+Nstars) ; n_block += 1
                    data['PartType0']['EnergySNCold'] = EnergySNCold[:Ngas]
                    data['PartType4']['EnergySNCold'] = EnergySNCold[Ngas:]

            if flags['flag_sn']: RateSN = read_quantity('f',2*N) ; RateSN.reshape(N,2) ; n_block += 1
            else: RateSN = np.zeros(2)
            
            try: info = read_info_block(n_block)
            except: 
                info = None
                print('Non-info; n_block={:d}'.format(n_block))

            eofi = np.fromfile(f,'b',-1)
            if eofi.size > 0: print('WARNING: EOF dont found', eofi.size)

    finally:
        if compress_file: os.remove(fname)
       
    # -------------- Escritura de control -------------- # 
    if control:
        print(" ")
        print("npart:"     , header['npart'])
        print("massarr:"   , header['massarr'])
        print("npartTotal:", header['npartTotal'])
        print("time:"      , header['time'])
        print("redshift:"  , header['redshift'])
        print("flags:"     , header['flag_sfr'], header['flag_feedback'], header['flag_cooling'])
        print("Nwithmass:" , Nwithmass)
        print(" ")
        print("Valores minimos y maximos de los siguintes parametros:")
        print("min  x, y, z" ,min(pos[:,0]),min(pos[:,1]),min(pos[:,2]))
        print("max  x, y, z" ,max(pos[:,0]),max(pos[:,1]),max(pos[:,2]))
        print("min vx,vy,vz" ,min(vel[:,0]),min(vel[:,1]),min(vel[:,2]))
        print("max vx,vy,vz" ,max(vel[:,0]),max(vel[:,1]),max(vel[:,2]))
        print("id"    ,min(id1) ,max(id1))
        print('masa'  ,min(mass),max(mass))
        if len(rho)  > 0: print("rho"   ,min(rho) ,max(rho))
        if len(hsml) > 0: print("Hsml"  ,min(hsml),max(hsml))
        if len(u)    > 0: print("u"     ,min(u)   ,max(u) )
        if len(Eg)   > 0: print("W"     ,min(Eg)  ,max(Eg),len(Eg))
        if len(zs)   > 0: print("zs"    ,min(zs)  ,max(zs))
        if len(Zm[:,0]) > 0: print("Zm(0)" ,min(Zm[:,0]),max(Zm[:,0]))
        if len(Zm[:,3]) > 0: print("Zm(3)" ,min(Zm[:,3]),max(Zm[:,3]))
        if len(Zm[:,4]) > 0: print("Zm(4)" ,min(Zm[:,4]),max(Zm[:,4]))
        if len(Zm[:,6]) > 0: print("Zm(6)" ,min(Zm[:,6]),max(Zm[:,6]))
        print(" ")
    
    # -------------------------------- Reasignacion de variables ----------------------- #

    def reasign(k, p, v, m, N, u=[], TypePart=0):
        s = np.sum(npart[0:TypePart])
        k = k[s:s+N]
        p = p[s:s+N]
        v = v[s:s+N]
        if len(u)>0: u = u[s:s+N] 
        M = massarr[TypePart]
        if M: m = np.full((N,1), M)
        else:
            s = np.sum(npart[0:TypePart] * ~massarr[0:TypePart].astype(bool))
            m = m[s:s+N].reshape(N,1)
        return k,p,v,m,u

    field_name = ['ID', 'Coordinates', 'Velocity', 'Mass']
    if flags['flag_outputpotential']: field_name += ['PotencialEnergy'] 
    if Ngas>0: 
        aux = reasign(id1, pos, vel, mass, Ngas, Pot, TypePart=0)
        for i,j in enumerate(field_name): data['PartType0'][j] = aux[i]
    if Nhalo>0: 
        aux = reasign(id1, pos, vel, mass, Nhalo, Pot, TypePart=1)
        for i,j in enumerate(field_name): data['PartType1'][j] = aux[i]
    if Ndisk>0: 
        aux = reasign(id1, pos, vel, mass, Ndisk, Pot, TypePart=2)       
        for i,j in enumerate(field_name): data['PartType2'][j] = aux[i]
    if Nbulge>0: 
        aux = reasign(id1, pos, vel, mass, Nbulge, Pot, TypePart=3)
        for i,j in enumerate(field_name): data['PartType3'][j] = aux[i]
    if Nstars>0: 
        aux = reasign(id1, pos, vel, mass, Nstars, Pot, TypePart=4)
        for i,j in enumerate(field_name): data['PartType4'][j] = aux[i]
    
    return header, data, info