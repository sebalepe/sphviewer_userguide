from sphviewer.tools import QuickView
import sphviewer as sph
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from sphviewer.tools import Blend
import matplotlib.image as mpimg
from sphviewer.tools import cmaps as cmp
import cv2
import glob
import imageio
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
import math
from sphviewer.tools import hsv_tools

import read_binary_snap as rd

from importlib import reload
reload(rd)

class SnapViewer: #guarda toda la informacion y facilita algunos elementos
    
    def __init__(self, path='', empty=False, flags=False):
        self.path = path
        self.empty = empty
        self.keys = ''
        self.header = None
        self.info = None
        self.flags = flags
        self.part = [] #tiene todos los datos con sus respectivos diccionarios
        self.pos = [] #todas las posiciones (por facilidad)
        self.mass = [] #todas las masas (por facilidad)
        self.vels = [] #todas las velocidades (por facilidad)
        
        self.init()
        
    def init(self): #guarda todos los datos tomando un path a un archivo en binario
        if not self.empty:
            if not self.flags:
                self.header, data, self.info = rd.read(self.path)
            else:
                self.header, data, self.info = rd.read(self.path, flags=self.flags)
                
            header_keys = f'Header: {list(self.header.keys())}\n\n'
            data_keys = list(data.keys())
            ptyp=[data[i] for i in data_keys]
            ptyp_keys = '\n\n'.join([f'Part Type {i} : {list(ptyp[i].keys())}' for i in range(len(data))])
            self.keys+= header_keys + ptyp_keys
            for i in range(len(ptyp)):
                self.part.append({})
                for u in list(ptyp[i].keys()):
                    self.part[i][f'{u}'] = ptyp[i][u]
                self.pos.append(ptyp[i]['Coordinates'])
                self.mass.append(ptyp[i]['Mass'])
                self.vels.append(ptyp[i]['Velocity'])
        else:
            print('Empty SnapViewer')
    
    ##Usefull Functions
    def get_keys(self): #imprime todas las llaves del archivo
        print(self.keys)
    
    def get_v(self, img, value_min, value_max): # cambia los valores vmin y vmax
        vmin = np.min(img)
        vmax = np.max(img)
        return vmin*value_min, vmax*value_max
    
    def interpol(self, t_inicio, t_final, frames): #interpola entre inicio y fin de una transicion en video
        x_i, y_i, z_i = t_inicio          #los frames indican en cuantos frames quiero llegar
        x_f, y_f, z_f = t_final
        x_new = np.linspace(x_i,x_f, frames)
        y_new = np.linspace(y_i,y_f, frames)
        z_new = np.linspace(z_i,z_f, frames)
        return x_new, y_new, z_new
    
    def centroid(self, pos, cosm=False, only_one=False): #encuentra el centro de masa/ velocidad centro de masa
        pos = pos[:]
        mass = self.mass[:]
        if not only_one:
            for i in [1,2]:
                pos.pop(i)
                mass.pop(i)
        if cosm:
            pos.pop(1)
            mass.pop(1)
        pos = np.concatenate(pos, axis=0)
        mass = np.concatenate(mass, axis=0)
        value = np.sum(pos * mass , axis=0)
        x,y,z = value / sum(mass)
        return x, -y, z
    
    def local_centroid(self, part, source, area):
        x, y, z = source

        l_x, l_y, l_z  = self.pos[part][:,0], self.pos[part][:,1], self.pos[part][:,2]

        d_x, d_y, d_z = np.abs(l_x - x), np.abs(l_y - y), np.abs(l_z - z)
        dis = np.sqrt(d_x ** 2 + d_y ** 2 + d_z ** 2)
        indexs = [i for i in range(len(dis)) if dis[i] <= area] 
        
        pos = np.array([self.pos[part][i] for i in indexs])
        mass = np.array([self.mass[part][i] for i in indexs])
        
        value = np.sum(pos * mass , axis=0)
        x,y,z = value / sum(mass)
        return x, -y, z

    def pos_id(self, part, _id): #toma el id de una particula y retorna su posicion
        index = np.where(self.part[part]['ID'] == _id)
        x,y,z = self.pos[part][index[0][0]]
        return x, -y, z
    
    def source_id(self, part, source):
        try:
            x,y,z = source
        except:
            x,y = source
        l_x = self.pos[part][:,0]
        l_y = self.pos[part][:,1]
        d_x = np.abs(l_x - x)
        d_y = np.abs(l_y - y)
        dis = d_x ** 2 + d_y ** 2
        i = np.argmin(dis)
        return self.part[part]['ID'][i]
    
    ## Funciones de SPH-Viewer
    def quickview(self, **kwargs):
        qv = QuickView(**kwargs)
        return qv
    
    def particles(self, pos, mass):
        particles = sph.Particles(pos, mass)
        return particles
    
    def scene(self, particles):
        scene = sph.Scene(particles)
        return scene
    
    def render(self, scene):
        render = sph.Render(scene)
        return render
    
    def blend(self, img1, img2):
        blend = Blend.Blend(img1, img2)
        return blend

    ## Plot Functions   
    def vfield_plot(self, ax,pos,mass,vel, extent, u, x, y, z, v_cm): #genera plot de streamlines de velocidad
        qv = QuickView(pos, r='infinity', mass=mass, x=x,y=y,z=z, extent=extent, plot=False, logscale=False)
        hsml = qv.get_hsml()
        density_field = qv.get_image()
        vfield = []
        mt = sum(mass)
        for i in range(2):
            qv1 = QuickView(pos, mass=(vel[:,i]- v_cm[i])*mt, hsml=hsml, r='infinity', x=x, y=y, z=z,
                           plot=False, extent=extent, logscale=False)
            vfield.append(qv1.get_image() / density_field)
        X = np.linspace(extent[0], extent[1], 500)
        Y = np.linspace(extent[2], extent[3], 500) 
        ax.imshow(np.log10(density_field), origin='lower', extent=extent, cmap='bone')
        v = np.log10(np.sqrt(vfield[0] ** 2 + vfield[1] ** 2))
        color = v / np.max(v)
        lw = color * 2
        streams = ax.streamplot(X, Y, vfield[0], vfield[1], color=color,
            density=1.5, cmap='jet', linewidth=lw, arrowsize=1)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.minorticks_on()
        ax.set_title(f'Part Type {u} QuickView')
        ax.set_xlabel(r'$\rm X / \ Mpc \ h^{-1}$', size=10)
        ax.set_ylabel(r'$\rm Y / \ Mpc \ h^{-1}$', size=10)
    
    def velocity_field(self, extent=[-15,15,-15,15]): #muestra campos de velocidad de todos los part types
        fig = plt.figure(figsize=(20, 20))
        gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[1,1,1])
        x,y,z = self.centroid(self.pos)
        v_cm = list(self.centroid(self.vels))
        axs = []
        for i in range(3):
            for u in range(2):
                ax=fig.add_subplot(gs[i, u])
                axs.append(ax)
        for i in range(len(self.pos)):
            self.vfield_plot(axs[i],self.pos[i], self.mass[i], self.vels[i], extent, i, x,y,z, v_cm)
        plt.show()

            
class SnapEvolution: #para leer multiples snaps
    
    def __init__(self, path, quantity, format_='', flags=False):
        self.path = path
        self.format = format_
        self.quantity = quantity
        self.snaps = []
        self.files = []
        self.flags = flags
        self.error = 0
        
        self.init()
    
    def init(self): #guarda todos los snaps
        for i in range(self.quantity):
            try:
                path = self.path + str('%03d'%i) + f'{self.format}'
                self.snaps.append(SnapViewer(path, flags=self.flags))
                self.files.append(self.path + str('%03d'%i) + f'{self.format}')
            except:
                self.error += 1
        if self.error > 0:
            print(f'error while adding {self.error} files')
        
        print(f'{len(self.snaps)} files added')
    
    def add_file(self, path):
        try:
            self.snaps.append(SnapViewer(path, flags=self.flags))
            self.files.append(path)
            print('File added')
        except:
            print('Path error')
                
    def see_files(self):
        if len(self.snaps) == 0:
            print('no files saved')
        for i in self.files:
            print(i)
    
    def get_keys(self):
        self.snaps[0].get_keys()
    
    def animate(self, iter): #funcion para animar streemlines de velocidad
        X = np.linspace(self.extent[0], self.extent[1], 500)
        Y = np.linspace(self.extent[2], self.extent[3], 500)
        x,y,z = self.snaps[iter].centroid(self.snaps[iter].pos)
        v_cm = self.snaps[iter].centroid(self.snaps[iter].vels)
        qv = QuickView(pos=self.snaps[iter].pos[self.part], r='infinity', mass=self.snaps[iter].mass[self.part], 
                       x=x,y=y,z=z, extent=self.extent, plot=False, logscale=False)
        hsml = qv.get_hsml()
        density_field = qv.get_image()
        vfield = []
        mt = sum(self.snaps[iter].mass[self.part])
        for i in range(2):
            qv1 = QuickView(pos=self.snaps[iter].pos[self.part], mass=(self.snaps[iter].vels[self.part][:,i]- v_cm[i])*mt, 
                            hsml=hsml, r='infinity', x=x,y=y,z=z, plot=False, extent=self.extent, logscale=False)
            vfield.append(qv1.get_image() / density_field)
        self.ax.collections = [] 
        self.ax.patches = [] 
        v = np.log10(np.sqrt(vfield[0] ** 2 + vfield[1] ** 2))
        color = v / np.max(v)
        lw = color * 2
        stream = self.ax.streamplot(X, Y, vfield[0], vfield[1], color=color,
            density=1.5, cmap='jet', linewidth=lw, arrowsize=1)
        self.ax.imshow(np.log10(density_field), origin='lower', extent=extent, cmap='bone')
        return stream
    
    def vfield_animation(self, part, frames, path, extent=[-15,15,-15,15]): #crea gif animado de evolucion de vfields
        self.part = part
        self.extent=extent
        
        x,y,z = self.snaps[0].centroid(self.snaps[0].pos)
        v_cm = self.snaps[0].centroid(self.snaps[0].vels)
        
        qv = QuickView(pos=self.snaps[0].pos[part], r='infinity', mass=self.snaps[0].mass[part], x=x, y=y, z=z, 
                       extent=extent, plot=False, logscale=False)
        hsml = qv.get_hsml()
        density_field = qv.get_image()
        vfield = []
        mt = sum(self.snaps[0].mass[part])
        for i in range(2):
            qv1 = QuickView(pos=self.snaps[0].pos[part], mass=(self.snaps[0].vels[part][:,i]- v_cm[i])*mt, 
                            hsml=hsml, r='infinity', x=x, y=y, z=z,plot=False, extent=extent, logscale=False)
            vfield.append(qv1.get_image() / density_field)

        fig, self.ax = plt.subplots()
        X = np.linspace(extent[0], extent[1], 500)
        Y = np.linspace(extent[2], extent[3], 500)
        v = np.log10(np.sqrt(vfield[0] ** 2 + vfield[1] ** 2))
        color = v / np.max(v)
        lw = color * 2
        stream = self.ax.streamplot(X, Y, vfield[0], vfield[1], color=color,
                density=1.5, cmap='jet', linewidth=lw, arrowsize=1)
        self.ax.imshow(np.log10(density_field), origin='lower', extent=extent, cmap='bone')
        anim =   animation.FuncAnimation(fig, self.animate, frames=frames, interval=50, blit=False, repeat=False)
        anim.save(f'{path}', writer='imagemagick', fps=60)
            
    def vmean(self, imgs, quantity):
        vmins = []
        vmaxs = []
        for i in range(quantity):
            img = imgs[i]
            vmins.append(np.min(img))
            img2 = imgs[::-1][i]
            vmaxs.append(np.max(img2))
        
        return np.mean(vmins), np.mean(vmaxs)
    
        
class Functions:
    
    def __init__(self):
        self.flags = {   
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

    def init_image(self): #inicio figura (para no abrir 200)
        self.fig = plt.figure(figsize=(8,8))
        self.gs = gridspec.GridSpec(1, 1)
    
    def end_image(self): #cierro figura
        plt.show()
            
    def all_functions(self):
        functions = [
        'all_functions()',
        'init_image()',
        'end_image()',
        """show_image(img, extent=None, vmin=None, vmax=None, cmap='magma', label='', l_color='white',
          hori=True, save=False, path='', xlabel='', ylabel='', show_cbar=True, dpi=100)""",
        """hsv_image(imgh=10, imgv=10, extent=None, vminh=None, vmaxh=None, vminv=None, vmaxv=None,
          rgbmin=0, rgbmax=1, xlabel='', ylabel='', xlabel_hsv='', ylabel_hsv='', l_color='w', 
          path='test', save=False, dpi=100)""",
        'save_gif(path1, path2)',
        'save_video(path1, path2, fps=15)',
        'transition(path_img1, path_img2, path_save, frames, index)',
        'cmap_from_image(path='', reverse=False):',
        "cmap_from_list(colors, bins=1000 ,name='my_cmap')",
        'fix_img(img, n=1.23)'
        ]
        [print(i+'\n') for i in functions]
        
    def show_image(self, img, extent=None, vmin=None, vmax=None, cmap='magma', label='', l_color='white',
                   hori=True, save=False, path='', xlabel='', ylabel='', show_cbar=True, dpi=100):
        
        if(vmin == None): vmin = np.min(img)
        if(vmax == None): vmax = np.max(img)
        
        # muestra una figura la colorbar dentro
        ax0 = plt.subplot(self.gs[0, 0])
        cm = plt.cm.get_cmap(cmap)
        if extent is None:
            plt.imshow(img, vmin=vmin, vmax=vmax, cmap=cm)
        else:
            plt.imshow(img, vmin=vmin, vmax=vmax, cmap=cm, extent=extent)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        if hori and show_cbar:
            cbaxes = inset_axes(ax0, width="40%", height="3%", loc=9) 
            cbar = plt.colorbar(cax=cbaxes, ticks=[0.,1], orientation='horizontal')
            cbar.set_label(label, color='white')
            lines1 = [i for i in np.linspace(vmin, vmax, 5)]
            lines2 = [format(i,'.1e') for i in np.linspace(vmin, vmax, 5)]
            cbar.set_ticks(lines1)
            cbar.ax.set_xticklabels(lines2, color=l_color, fontsize=7)
            
        if not hori and show_cbar:
            cbaxes = inset_axes(ax0, width="3%", height="40%", loc=6)
            cbar = plt.colorbar(cax=cbaxes, ticks=[0.,1], orientation='vertical')
            cbar.set_label(label, color=l_color)
            lines1 = [i for i in np.linspace(vmin, vmax, 5)]
            lines2 = [format(i,'.1e') for i in np.linspace(vmin, vmax, 5)]
            cbar.set_ticks(lines1)
            cbar.ax.set_yticklabels(lines2, color=l_color, fontsize=7)
        
        if save:
            plt.savefig(f'{path}.png', dpi=dpi)

    def hsv_image(self, imgh=10, imgv=10, extent=None, vminh=None, vmaxh=None, vminv=None, vmaxv=None,
                  rgbmin=0, rgbmax=1, xlabel='', ylabel='', xlabel_hsv='', ylabel_hsv='', l_color='w', 
                  path='test', save=False, dpi=100):
        '''
        imgh: Image colored by the horizontal part of the cmap. Follow a list of colors.
imgv: Image colored by the vertical part of the cmap. Goes from black to white.
extent: The extent of the image, None by default.
vminh: The vmin value for the imgh, np.min(imgh) by default.
vmaxh: The vmax value for the imgh, np.min(imgh) by default.
vminv: The vmin value for the imgv, np.min(imgv) by default.
vmaxv: The vmax value for the imgv, np.min(imgv) by default.
rgbmin: The min value for the colorlist. Change where it starts. 0 by default.
rgbmax: The max value for the colorlist. Change where it ends. 1 by default.
xlabel, ylabel: labels for the x and y axes.
xlabel_hsv, ylabel_hsv: labels for the x and y axes of the hsv colobar.
l_color: color of the labels on the hsv colorbar. White by default.
path, save, dpi=100: Parameters to save the image.
        '''
        
        if(vminh == None): vminh = np.min(imgh)
        if(vmaxh == None): vmaxh = np.max(imgh)
        if(vminv == None): vminv = np.min(imgv)
        if(vmaxv == None): vmaxv = np.max(imgv)
    
        def hsv_colorbar(axs, minh=0, maxh=1, minv=0, maxv=1,
                     pos=(0.8,0.8,0.15,0.15), color='w', hmin=0, hmax=1):

            cbh = np.tile(np.linspace(minh,maxh,500),(500,1)).astype(np.float32)
            cbv = np.tile(np.linspace(minv,maxv,500).reshape(-1,1),(1,500)).astype(np.float32)
            rgb = hsv_tools.image_from_hsv(h=cbh, v=cbv, img_hmin=minh, img_hmax=maxh, img_vmin=minv, img_vmax=maxv,
                                          hmin=hmin, hmax=hmax)
            axs = axs.inset_axes(pos)
            axs.imshow(rgb, origin='lower', extent=(minh,maxh,minv,maxv), aspect='auto')

            #axs.spines[:].set_color(color) #me tira error ):
            [axs.spines[key].set_color(color) for key in axs.spines.keys()]
            axs.tick_params(axis='x', colors=color, labelrotation=30)
            axs.tick_params(axis='y', colors=color)
            
            linesh = [i for i in np.linspace(minh, maxh, 3)]
            linesv = [i for i in np.linspace(minv, maxv, 3)]
            axs.set_xticks(linesh)
            axs.set_yticks(linesv)
             
            linesx = [np.format_float_scientific(i, precision = 1, exp_digits=1) for i in np.linspace(minh, maxh, 3)]
            linesy = [np.format_float_scientific(i, precision = 1, exp_digits=1)for i in np.linspace(minv, maxv, 3)]
            axs.set_xticklabels(linesx, color=l_color, fontsize=10)
            axs.set_yticklabels(linesy, color=l_color, fontsize=10)

            return axs

        ax0 = plt.subplot(self.gs[0, 0])
        rgb = hsv_tools.image_from_hsv(h=imgh, v=imgv, img_hmin=vminh, img_hmax=vmaxh, img_vmin=vminv, img_vmax=vmaxv,
                                   hmin=rgbmin, hmax=rgbmax)
        if extent is None:
            ax0.imshow(rgb)
        else:
            ax0.imshow(rgb, extent=extent)
        cb = hsv_colorbar(ax0, minh=vminh, maxh=vmaxh, minv=vminv, maxv=vmaxv, hmin=rgbmin, hmax=rgbmax, color=l_color)
        cb.set_xlabel(xlabel_hsv, color=l_color)
        cb.set_ylabel(ylabel_hsv, color=l_color)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        if save:
            plt.savefig(f'{path}.png', dpi=dpi)

    def save_gif(self, path1, path2): #une las imagenes de una carpeta (ordenados) en forma de gif
        img_array = []
        for filename in glob.glob(f'{path1}*.png'):
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width,height)
            img_array.append(img)
        with imageio.get_writer(f"{path2}.gif", mode="I") as writer:
            for idx, frame in enumerate(img_array):
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                writer.append_data(rgb_frame)

    def save_video(self, path1, path2, fps=15): #une las imagenes de una carpeta (ordenados) en forma de video
        img_array = []
        for filename in glob.glob(f'{path1}*.png'):
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width,height)
            img_array.append(img)
        out = cv2.VideoWriter(f'{path2}.mp4',cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()

    def transition(self, path_img1, path_img2, path_save, frames, index):

        imgcv1 = cv2.imread(path_img1) #abro las imagenes con opencv
        imgcv2 = cv2.imread(path_img2)
        rgb1 = cv2.cvtColor(imgcv1, cv2.COLOR_BGR2RGB) #arreglo el color
        rgb2 = cv2.cvtColor(imgcv2, cv2.COLOR_BGR2RGB)
        x,y,z = imgcv1.shape
        value = x/frames 
        for i in range(frames):
            value2 = value * i 
            max_ =  value + value2 
            capa = rgb1[0:int(max_), 0:x]
            rgb2[0:int(max_), 0:x] = capa
            u = index + i
            plt.imsave(f'{path_save}_' + str('%03d.png'%u), rgb2, dpi = 100)

    def cmap_from_image(self, path='', reverse=False):
        img = imread(path)
        colors_from_img = img[:, 0, :]
        if reverse:
            colors_from_img = colors_from_img[::-1]
        my_cmap = LinearSegmentedColormap.from_list('my_cmap', colors_from_img, N=280)
        return my_cmap

    def cmap_from_list(self, colors, bins=1000 ,name='my_cmap'):
        cmap = LinearSegmentedColormap.from_list(name, colors, N=bins)
        return cmap

    def fix_img(self, img, n=1.23):
        img1 = np.where(np.isnan(img), n, img)
        img2 = np.where(img1==-math.inf, n, img1)
        img3 = np.where(img2==n, np.amin(img2), img2)
        img4 = np.where(img3==math.inf, n, img3)
        img5 = np.where(img4==n, np.amax(img4), img4)
        return img5
 
   