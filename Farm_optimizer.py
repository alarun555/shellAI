# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 03:21:25 2020

@author: Arun
"""


# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 20:57:47 2020

@author: Arun
"""


# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 16:49:52 2020

@author: Arun
"""


# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 06:13:01 2020

@author: Arun
"""


# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 20:25:02 2020

@author: Arun
"""


# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 07:40:42 2020

@author: Arun
"""


# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 15:32:16 2020

@author: Arun
"""


import simanneal


# Module List
import numpy  as np
import pandas as pd                     
from   math   import radians as DegToRad       # Degrees to radians Conversion
import math
import random
import matplotlib.pyplot as plt

from shapely.geometry import Point             # Imported for constraint checking
from shapely.geometry.polygon import Polygon

import warnings
warnings.filterwarnings("ignore")

def getTurbLoc(turb_loc_file_name):
    """ 
    -**-THIS FUNCTION SHOULD NOT BE MODIFIED-**-
    
    Returns x,y turbine coordinates
    
    :Called from
        main function
    
    :param
        turb_loc_file_name - Turbine Loc csv file location
        
    :return
        2D array
    """
    
    df = pd.read_csv(turb_loc_file_name, sep=',', dtype = np.float32)
    turb_coords = df.to_numpy(dtype = np.float32)
    return(turb_coords)


def loadPowerCurve(power_curve_file_name):
    """
    -**-THIS FUNCTION SHOULD NOT BE MODIFIED-**-
    
    Returns a 2D numpy array with information about
    turbine thrust coeffecient and power curve of the 
    turbine for given wind speed
    
    :called_from
        main function
    
    :param
        power_curve_file_name - power curve csv file location
        
    :return
        Returns a 2D numpy array with cols Wind Speed (m/s), 
        Thrust Coeffecient (non dimensional), Power (MW)
    """
    powerCurve = pd.read_csv(power_curve_file_name, sep=',', dtype = np.float32)
    powerCurve = powerCurve.to_numpy(dtype = np.float32)
    return(powerCurve)
    

def binWindResourceData(wind_data_file_name):
    r"""
    -**-THIS FUNCTION SHOULD NOT BE MODIFIED-**-
    
    Loads the wind data. Returns a 2D array with shape (36,15). 
    Each cell in  array is a wind direction and speed 'instance'. 
    Values in a cell correspond to probability of instance
    occurence.  
    
    :Called from
        main function
        
    :param
        wind_data_file_name - Wind Resource csv file  
        
    :return
        1-D flattened array of the 2-D array shown below. Values 
        inside cells, rough probabilities of wind instance occurence. 
        Along: Row-direction (drct), Column-Speed (s). Array flattened
        for vectorization purpose. 
        
                      |0<=s<2|2<=s<4| ...  |26<=s<28|28<=s<30|
        |_____________|______|______|______|________|________|
        | drct = 360  |  --  |  --  |  --  |   --   |   --   |
        | drct = 10   |  --  |  --  |  --  |   --   |   --   |
        | drct = 20   |  --  |  --  |  --  |   --   |   --   |
        |   ....      |  --  |  --  |  --  |   --   |   --   |
        | drct = 340  |  --  |  --  |  --  |   --   |   --   |
        | drct = 350  |  --  |  --  |  --  |   --   |   --   |        
    """
    
    # Load wind data. Then, extracts the 'drct', 'sped' columns
    df = pd.read_csv(wind_data_file_name)
    wind_resource = df[['drct', 'sped']].to_numpy(dtype = np.float32)
    
    # direction 'slices' in degrees
    slices_drct   = np.roll(np.arange(10, 361, 10, dtype=np.float32), 1)
    ## slices_drct   = [360, 10.0, 20.0.......340, 350]
    n_slices_drct = slices_drct.shape[0]
    
    # speed 'slices'
    slices_sped   = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 
                        18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0]
    n_slices_sped = len(slices_sped)-1

    
    # placeholder for binned wind
    binned_wind = np.zeros((n_slices_drct, n_slices_sped), 
                           dtype = np.float32)
    
    # 'trap' data points inside the bins. 
    for i in range(n_slices_drct):
        for j in range(n_slices_sped):     
            
            # because we already have drct in the multiples of 10
            foo = wind_resource[(wind_resource[:,0] == slices_drct[i])] 

            foo = foo[(foo[:,1] >= slices_sped[j]) 
                          & (foo[:,1] <  slices_sped[j+1])]
            
            binned_wind[i,j] = foo.shape[0]  
    
    wind_inst_freq   = binned_wind/np.sum(binned_wind)
    wind_inst_freq   = wind_inst_freq.ravel()
    
    return(wind_inst_freq)


def searchSorted(lookup, sample_array):
    """
    -**-THIS FUNCTION SHOULD NOT BE MODIFIED-**-
    
    Returns lookup indices for closest values w.r.t sample_array elements
    
    :called_from
        preProcessing, getAEP
    
    :param
        lookup       - The lookup array
        sample_array - Array, whose elements need to be matched
                       against lookup elements. 
        
    :return
        lookup indices for closest values w.r.t sample_array elements 
    """
    lookup_middles = lookup[1:] - np.diff(lookup.astype('f'))/2
    idx1 = np.searchsorted(lookup_middles, sample_array)
    indices = np.arange(lookup.shape[0])[idx1]
    return indices

   

def preProcessing(power_curve):
    """
    -**-THIS FUNCTION SHOULD NOT BE MODIFIED-**-
    
    Doing preprocessing to avoid the same repeating calculations.
    Record the required data for calculations. Do that once.
    Data are set up (shaped) to assist vectorization. Used later in
    function totalAEP. 
    
    :called_from
        main function
    
    :param
        power_curve - 2D numpy array with cols Wind Speed (m/s), 
                      Thrust Coeffecient (non dimensional), Power (MW)
        
    :return
        n_wind_instances  - number of wind instances (int)
        cos_dir           - For coordinate transformation 
                            2D Array. Shape (n_wind_instances,1)
        sin_dir           - For coordinate transformation 
                            2D Array. Shape (n_wind_instances,1)
        wind_sped_stacked - column staked all speed instances n_turb times. 
        C_t               - 3D array with shape (n_wind_instances, n_turbs, n_turbs)
                            Value changing only along axis=0. C_t, thrust coeff.
                            values for all speed instances. 
    """
    # number of turbines
    n_turbs       =   50
    
    # direction 'slices' in degrees
    slices_drct   = np.roll(np.arange(10, 361, 10, dtype=np.float32), 1)
    ## slices_drct   = [360, 10.0, 20.0.......340, 350]
    n_slices_drct = slices_drct.shape[0]
    
    # speed 'slices'
    slices_sped   = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 
                        18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0]
    n_slices_sped = len(slices_sped)-1
    
    # number of wind instances
    n_wind_instances = (n_slices_drct)*(n_slices_sped)
    
    # Create wind instances. There are two columns in the wind instance array
    # First Column - Wind Speed. Second Column - Wind Direction
    # Shape of wind_instances (n_wind_instances,2). 
    # Values [1.,360.],[3.,360.],[5.,360.]...[25.,350.],[27.,350.],29.,350.]
    wind_instances = np.zeros((n_wind_instances,2), dtype=np.float32)
    counter = 0
    for i in range(n_slices_drct):
        for j in range(n_slices_sped): 
            
            wind_drct =  slices_drct[i]
            wind_sped = (slices_sped[j] + slices_sped[j+1])/2
            
            wind_instances[counter,0] = wind_sped
            wind_instances[counter,1] = wind_drct
            counter += 1

	# So that the wind flow direction aligns with the +ve x-axis.			
    # Convert inflow wind direction from degrees to radians
    wind_drcts =  np.radians(wind_instances[:,1] - 90)
    # For coordinate transformation 
    cos_dir = np.cos(wind_drcts).reshape(n_wind_instances,1)
    sin_dir = np.sin(wind_drcts).reshape(n_wind_instances,1)
    
    # create copies of n_wind_instances wind speeds from wind_instances
    wind_sped_stacked = np.column_stack([wind_instances[:,0]]*n_turbs)
   
    # Pre-prepare matrix with stored thrust coeffecient C_t values for 
    # n_wind_instances shape (n_wind_instances, n_turbs, n_turbs). 
    # Value changing only along axis=0. C_t, thrust coeff. values for all 
    # speed instances.
    # we use power_curve data as look up to estimate the thrust coeff.
    # of the turbine for the corresponding closest matching wind speed
    indices = searchSorted(power_curve[:,0], wind_instances[:,0])
    C_t     = power_curve[indices,1]
    # stacking and reshaping to assist vectorization
    C_t     = np.column_stack([C_t]*(n_turbs*n_turbs))
    C_t     = C_t.reshape(n_wind_instances, n_turbs, n_turbs)
    
    return(n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t)


def getAEP(turb_rad, turb_coords, power_curve, wind_inst_freq, 
            n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t):
    
    """
    -**-THIS FUNCTION SHOULD NOT BE MODIFIED-**-
    
    Calculates AEP of the wind farm. Vectorised version.
    
    :called from
        main
        
    :param
        turb_diam         - Radius of the turbine (m)
        turb_coords       - 2D array turbine euclidean x,y coordinates
        power_curve       - For estimating power. 
        wind_inst_freq    - 1-D flattened with rough probabilities of 
                            wind instance occurence.
                            n_wind_instances  - number of wind instances (int)
        cos_dir           - For coordinate transformation 
                            2D Array. Shape (n_wind_instances,1)
        sin_dir           - For coordinate transformation 
                            2D Array. Shape (n_wind_instances,1)
        wind_sped_stacked - column staked all speed instances n_turb times. 
        C_t               - 3D array with shape (n_wind_instances, n_turbs, n_turbs)
                            Value changing only along axis=0. C_t, thrust coeff.
                            values for all speed instances. 
    
    :return
        wind farm AEP in Gigawatt Hours, GWh (float)
    """
    # number of turbines
    n_turbs        =   turb_coords.shape[0]
    assert n_turbs ==  50, "Error! Number of turbines is not 50."
    
    # Prepare the rotated coordinates wrt the wind direction i.e downwind(x) & crosswind(y) 
    # coordinates wrt to the wind direction for each direction in wind_instances array
    rotate_coords   =  np.zeros((n_wind_instances, n_turbs, 2), dtype=np.float32)
    # Coordinate Transformation. Rotate coordinates to downwind, crosswind coordinates
    rotate_coords[:,:,0] =  np.matmul(cos_dir, np.transpose(turb_coords[:,0].reshape(n_turbs,1))) - \
                           np.matmul(sin_dir, np.transpose(turb_coords[:,1].reshape(n_turbs,1)))
    rotate_coords[:,:,1] =  np.matmul(sin_dir, np.transpose(turb_coords[:,0].reshape(n_turbs,1))) +\
                           np.matmul(cos_dir, np.transpose(turb_coords[:,1].reshape(n_turbs,1)))
 
    
    # x_dist - x dist between turbine pairs wrt downwind/crosswind coordinates)
    # for each wind instance
    x_dist = np.zeros((n_wind_instances,n_turbs,n_turbs), dtype=np.float32)
    for i in range(n_wind_instances):
        tmp = rotate_coords[i,:,0].repeat(n_turbs).reshape(n_turbs, n_turbs)
        x_dist[i] = tmp - tmp.transpose()
    

    # y_dist - y dist between turbine pairs wrt downwind/crosswind coordinates)
    # for each wind instance    
    y_dist = np.zeros((n_wind_instances,n_turbs,n_turbs), dtype=np.float32)
    for i in range(n_wind_instances):
        tmp = rotate_coords[i,:,1].repeat(n_turbs).reshape(n_turbs, n_turbs)
        y_dist[i] = tmp - tmp.transpose()
    y_dist = np.abs(y_dist) 
     

    # Now use element wise operations to calculate speed deficit.
    # kw, wake decay constant presetted to 0.05
    # use the jensen's model formula. 
    # no wake effect of turbine on itself. either j not an upstream or wake 
    # not happening on i because its outside of the wake region of j
    # For some values of x_dist here RuntimeWarning: divide by zero may occur
    # That occurs for negative x_dist. Those we anyway mark as zeros. 
    sped_deficit = (1-np.sqrt(1-C_t))*((turb_rad/(turb_rad + 0.05*x_dist))**2) 
    sped_deficit[((x_dist <= 0) | ((x_dist > 0) & (y_dist > (turb_rad + 0.05*x_dist))))] = 0.0
    
    
    # Calculate Total speed deficit from all upstream turbs, using sqrt of sum of sqrs
    sped_deficit_eff  = np.sqrt(np.sum(np.square(sped_deficit), axis = 2))

    
    # Element wise multiply the above with (1- sped_deficit_eff) to get
    # effective windspeed due to the happening wake
    wind_sped_eff     = wind_sped_stacked*(1.0-sped_deficit_eff)

    
    # Estimate power from power_curve look up for wind_sped_eff
    indices = searchSorted(power_curve[:,0], wind_sped_eff.ravel())
    power   = power_curve[indices,2]
    power   = power.reshape(n_wind_instances,n_turbs)
    
    # Farm power for single wind instance 
    power   = np.sum(power, axis=1)
    
    # multiply the respective values with the wind instance probabilities 
    # year_hours = 8760.0
    AEP = 8760.0*np.sum(power*wind_inst_freq)
    
    # Convert MWh to GWh
    AEP = AEP/1e3
    
    return(AEP)
    

    
def checkConstraints(turb_coords, turb_diam):
    """
    -**-THIS FUNCTION SHOULD NOT BE MODIFIED-**-
    
    Checks if the turbine configuration satisfies the two
    constraints:(i) perimeter constraint,(ii) proximity constraint 
    Prints which constraints are violated if any. Note that this 
    function does not quantifies the amount by which the constraints 
    are violated if any. 
    
    :called from
        main 
        
    :param
        turb_coords - 2d np array containing turbine x,y coordinates
        turb_diam   - Diameter of the turbine (m)
    
    :return
        None. Prints messages.   
    """
    bound_clrnc      = 50
    prox_constr_viol = False
    peri_constr_viol = False
    
    # create a shapely polygon object of the wind farm
    farm_peri = [(0, 0), (0, 4000), (4000, 4000), (4000, 0)]
    farm_poly = Polygon(farm_peri)
    
    # checks if for every turbine perimeter constraint is satisfied. 
    # breaks out if False anywhere
    for turb in turb_coords:
        turb = Point(turb)
        inside_farm   = farm_poly.contains(turb)
        correct_clrnc = farm_poly.boundary.distance(turb) >= bound_clrnc
        if (inside_farm == False or correct_clrnc == False):
            peri_constr_viol = True
            break
    
    # checks if for every turbines proximity constraint is satisfied. 
    # breaks out if False anywhere
    for i,turb1 in enumerate(turb_coords):
        for turb2 in np.delete(turb_coords, i, axis=0):
            if  np.linalg.norm(turb1 - turb2) < 4*turb_diam:
                prox_constr_viol = True
                break
    
    # print messages
    if  peri_constr_viol  == True  and prox_constr_viol == True:
          print('Somewhere both perimeter constraint and proximity constraint are violated\n')
    elif peri_constr_viol == True  and prox_constr_viol == False:
          print('Somewhere perimeter constraint is violated\n')
    elif peri_constr_viol == False and prox_constr_viol == True:
          print('Somewhere proximity constraint is violated\n')
    else: print('Both perimeter and proximity constraints are satisfied !!\n')
        
    return()

if __name__ == "__main__":

    # Turbine Specifications.
    # -**-SHOULD NOT BE MODIFIED-**-
    turb_specs    =  {   
                         'Name': 'Anon Name',
                         'Vendor': 'Anon Vendor',
                         'Type': 'Anon Type',
                         'Dia (m)': 100,
                         'Rotor Area (m2)': 7853,
                         'Hub Height (m)': 100,
                         'Cut-in Wind Speed (m/s)': 3.5,
                         'Cut-out Wind Speed (m/s)': 25,
                         'Rated Wind Speed (m/s)': 15,
                         'Rated Power (MW)': 3
                     }
    turb_diam      =  turb_specs['Dia (m)']
    turb_rad       =  turb_diam/2 
    
    # Turbine x,y coordinates
    turb_coords   =  getTurbLoc(r'..\Shell_Hackathon Dataset\turbine_loc_test.csv')
    
    # Load the power curve
    power_curve   =  loadPowerCurve('..\Shell_Hackathon Dataset\power_curve.csv')
    
    # Pass wind data csv file location to function binWindResourceData.
    # Retrieve probabilities of wind instance occurence.
    wind_inst_freq =  binWindResourceData(r'..\Shell_Hackathon Dataset\Wind Data\wind_data_2007.csv')   
    
    # Doing preprocessing to avoid the same repeating calculations. Record 
    # the required data for calculations. Do that once. Data are set up (shaped)
    # to assist vectorization. Used later in function totalAEP.
    n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t = preProcessing(power_curve)
    
    # check if there is any constraint is violated before we do anything. Comment 
    # out the function call to checkConstraints below if you desire. Note that 
    # this is just a check and the function does not quantifies the amount by 
    # which the constraints are violated if any. 
    checkConstraints(turb_coords, turb_diam)
    
    print('Calculating AEP......')
    AEP = getAEP(turb_rad, turb_coords, power_curve, wind_inst_freq, 
                  n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t) 
    print('Total power produced by the wind farm is: ', "%.12f"%(AEP), 'GWh')
    
"simulated Annealing starts"

counter=0

class BinaryAnnealer(simanneal.Annealer):
    
    
    Tmax = 100000 #Starting Anneal Temp-Can be changed def=25000
    Tmin = 99990#End Anneal Temp-Can be changed def=2.5
    steps = 5000 #Number of iterations- can be changed
    
    

    def move(self):
        
        "choose a random index from the matrix"
        #i = random.randrange(self.state.size)
        #i=random.randint(0, 899)
        
        #print("size of 1d array is")
        #print(self.state.size)
        #print("Random Index is:")
        #print(i)
        
        ##store if value at i is 0 or 1
        #z=self.state.flat[i]
        
        "Counter for plotting"
        #global counter
        z=1000
        while(z!=1 and z!=0):
            i=random.randint(0, 62499)
            #i = random.randrange(self.state.size)
            z=int(self.state.flat[i])
            #print(z)
        
        #check if chosen value is 0 or 1 else recursively repeat loop
        if(z==1 or z==0):
            #counter+=1 #counter for plotting
            #print("z now is")
            #print(z)
            if(z==1):
                #print("Random Index is inside:")
                #print(i)
                self.state.flat[i]=0 #switch 1 to 0
                #split linear index i into 2d indices bj and bi
                bj=int(i//250)
                bi=int(i%250)
                for v in range(bj-24,bj+25): #switch neighbours by adding one from their -ve previous vals
                    for u in range(bi-24,bi+25):
                        if(v<0 or v>249 or u<0 or u>249):
                            continue
                        dist=math.sqrt((((v-bj)*16)**2)+(((u-bi)*16)**2)) #distance between cell and turbine at center
                        if(dist>=400):
                            continue
                        if(self.state.flat[(v*250)+u]!=0): #condition shields center new 0
                            self.state.flat[(v*250)+u]+=1
                #now switch a 0 to 1 for balance
                #x = self.state.flat #save the flat array into x
                zero_idx=np.argwhere(self.state.flat==0) #finding and storing indices of elements 0
                np.random.shuffle(zero_idx)
                indexZ=zero_idx[0][0]
                self.state.flat[indexZ]=1
                #negate -1 from boundaries of new 1
                bj=int(indexZ//250)
                bi=int(indexZ%250)
                for v in range(bj-24,bj+25): 
                    for u in range(bi-24,bi+25):
                        if(v<0 or v>249 or u<0 or u>249):
                            continue
                        dist=math.sqrt((((v-bj)*16)**2)+(((u-bi)*16)**2)) #distance between cell and turbine at center
                        if(dist>=400):
                            continue
                        if(self.state.flat[(v*250)+u]!=1): #condition shields center
                            self.state.flat[(v*250)+u]-=1
                #self.state.flat=x
                """
                #viz new state
                if(counter>1):         #counter==100 or counter==200 or counter==300 or    
                    vizmatrix=np.zeros((250,250))
                    for i in range(0,250):
                        for j in range(0,250):
                            vizmatrix[i][j]=self.state.flat[i*250+j]
                    #plot the matrix
                    fig, ax = plt.subplots()
                    min_val, max_val = -6, 1
                    ax.matshow(vizmatrix, cmap=plt.cm.Blues)
                    
                    for i in range(0,250):
                        for j in range(0,250):
                            c = vizmatrix[j][i]
                            ax.text(i, j, int(c))#, va='center', ha='center')
                    
                    plt.show()
                """
                """
                #alternate plotting method
                vizmatrix=np.zeros((250,250))
                for j in range(0,250):
                    for i in range(0,250):
                        vizmatrix[j][i]=self.state.flat[j*250+i]
                print(pd.DataFrame(vizmatrix))
                plt.matshow(vizmatrix);
                plt.colorbar()
                plt.show()
                """
                
            else: #z==0
                #switch 0 to 1
                #print("Random Index is inside:")
                #print(i)
                #counter+=1 #counter for plotting
                self.state.flat[i]=1
                bj=int(i//250)
                bi=int(i%250)
                for v in range(bj-24,bj+25):
                    for u in range(bi-24,bi+25):
                        if(v<0 or v>249 or u<0 or u>249):
                            continue
                        dist=math.sqrt((((v-bj)*16)**2)+(((u-bi)*16)**2)) #distance between cell and turbine at center
                        if(dist>=400):
                            continue ##if distance b/w current cell and center >400 skip
                        if(self.state.flat[(v*250)+u]!=1):
                            self.state.flat[(v*250)+u]-=1
                #now switch random 1 to 0 for balance
                #x = self.state.flat
                zero_idx=np.argwhere(self.state.flat==1)
                np.random.shuffle(zero_idx)
                indexZ=zero_idx[0][0]
                self.state.flat[indexZ]=0
                #add 1 to boundaries of new 0
                bj=int(indexZ//250)
                bi=int(indexZ%250)
                for v in range(bj-24,bj+25): 
                    for u in range(bi-24,bi+25):
                        if(v<0 or v>249 or u<0 or u>249):
                            continue
                        dist=math.sqrt((((v-bj)*16)**2)+(((u-bi)*16)**2)) #distance between cell and turbine at center
                        if(dist>=400):
                            continue #for skipping some boundary points greater than 400 in the square proximity box
                        if(self.state.flat[(v*250)+u]!=0): #condition shields center
                            self.state.flat[(v*250)+u]+=1
                #self.state.flat=x
                """
                #visualizing the new state
                if(counter>1): #counter==100 or counter==200 or counter==300 or 
                    vizmatrix=np.zeros((250,250))
                    for i in range(0,250):
                        for j in range(0,250):
                            vizmatrix[i][j]=self.state.flat[i*250+j]
                    fig, ax = plt.subplots()
                    min_val, max_val = -6, 1
                    ax.matshow(vizmatrix, cmap=plt.cm.Blues)
                    
                    for i in range(0,250):
                        for j in range(0,250):
                            c = vizmatrix[j][i]
                            ax.text(i, j, int(c))#, va='center', ha='center')
                    
                    plt.show()
                """
                """
                vizmatrix=np.zeros((250,250))
                for j in range(0,250):
                    for i in range(0,250):
                        vizmatrix[j][i]=self.state.flat[j*250+i]
                print(pd.DataFrame(vizmatrix))
                plt.matshow(vizmatrix);
                plt.colorbar()
                plt.show()
                """
                
        #else:
            #self.move() #check if function continues after this recursive statement
            #return 0
        """
        #choose a random entry in the matrix
        i = random.randrange(self.state.size)
        
        #store if value at i is 0 or 1
        z=self.state.flat[i]
        #flip the entry at i
        self.state.flat[i] = 1 - self.state.flat[i]
        
        #if flipped value is a 1
        
        if(z==1):
            x = self.state.flat #save the flat array into x
            zero_idx=np.argwhere(x!=1) #finding and st
            np.random.shuffle(zero_idx)
            indexZ=zero_idx[0][0]
            x[indexZ]=1
            self.state.flat=x
        else: #if flipped is a 0
            x = self.state.flat
            non_zero_idx=np.argwhere(x!=0)
            np.random.shuffle(non_zero_idx)
            indexNZ=non_zero_idx[0][0]
            x[indexNZ]=0
            self.state.flat=x
         """   

    def energy(self):
        TurbCoords=np.zeros((50,2))
        
        s=0
        
        for j in range(0,250):
            for i in range(0,250):
                index=j*250+i
                if(self.state.flat[index]==1): #CONCENTRATE!!!
                    TurbCoords[s][0]=(i*16)+8
                    TurbCoords[s][1]=(j*16)+8
                    #print(TurbCoords[s])
                    s=s+1
                    
                    
                    """
                    if((j*80)+40<50):#pushing from 40m at boundary to 50m
                        TurbCoords[s][0]=50
                        TurbCoords[s][1]=(i*80)+40
                        s=s+1
                    elif((i*80)+40<50):
                        TurbCoords[s][0]=(j*80)+40
                        TurbCoords[s][1]=50
                        s=s+1
                    elif((j*80)+40>3950):
                        TurbCoords[s][0]=3950
                        TurbCoords[s][1]=(i*80)+40
                        s=s+1
                    elif((i*80)+40>3950):
                        TurbCoords[s][0]=(j*80)+40
                        TurbCoords[s][1]=3950
                        s=s+1
                    else:
                        TurbCoords[s][0]=(j*80)+40
                        TurbCoords[s][1]=(i*80)+40
                        #print(TurbCoords[s])
                        s=s+1
                    #if(s==50):
                        #break        
                    """
        """            
        vizuArray=np.zeros((250,250))
        for i in range(0,250):
            for j in range(0,250):
                vizuArray[i][j]=self.state.flat[i*250+j]
        print("Showing current matrix")
        plt.matshow(vizuArray);
        plt.colorbar()
        plt.show()
        """
        #print("Current AEP is:")
        #print(getAEP(turb_rad, TurbCoords, power_curve, wind_inst_freq, 
            #n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t))
        """
        #plot of each layout
        x = TurbCoords[:,0]
        y = TurbCoords[:,1]
        colors = (0,0,0)
        area = np.pi*3
        
        # Plot
        plt.scatter(x, y, s=area, c=colors, alpha=0.5)
        plt.title('Scatter plot Turbine Layout')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
        """
        
        #return -AEP value as energy function
        #VERY IMPORTANT: RISE AEP TO POWERS TO GIVE IMPORTANCE TO IT
        AEP_power_weight=100 #can be varied, 10 seems ideal
        return -getAEP(turb_rad, TurbCoords, power_curve, wind_inst_freq, 
            n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t)**AEP_power_weight
        
        
        # evaluate the function to minimize
        #return -np.linalg.det(self.state)
        
    print("\n")

#matrix = np.zeros((10, 10))
"""
#best manual iteration layout
matrixR=np.array([[1,1,1,1,1,1,1,1,1,1],
                 [1,0,0,0,0,0,0,0,0,1],
                 [1,0,0,0,0,0,0,0,0,1],
                 [1,1,1,1,0,1,1,1,1,1],
                 [1,0,0,0,0,0,0,0,0,1],
                 [1,0,0,0,0,0,0,0,0,1],
                 [1,1,1,1,0,1,1,1,1,1],
                 [1,0,0,0,0,0,0,0,0,1],
                 [1,0,0,0,0,0,0,0,0,1],
                 [1,1,1,1,1,1,1,1,1,1]])

"""
"""
#starter AEP 506 1000iterations
matrixR=np.array([[1,1,1,1,1,1,1,1,1,1],
                 [1,1,1,1,1,1,1,1,1,1],
                 [1,1,1,1,1,1,1,1,1,1],
                 [1,1,1,1,1,1,1,1,1,1],
                 [1,1,1,1,1,1,1,1,1,1],
                 [0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0]])
"""
"""
#starter AEP 504.04 1000iterations
matrix=np.array([[0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0],
                 [1,1,1,1,1,1,1,1,1,1],
                 [1,1,1,1,1,1,1,1,1,1],
                 [1,1,1,1,1,1,1,1,1,1],
                 [1,1,1,1,1,1,1,1,1,1],
                 [1,1,1,1,1,1,1,1,1,1]])
"""
"""
#starter AEP 505.57 1000iterations

matrixR=np.array([[0,0,0,0,0,0,0,0,0,0],
                 [1,1,1,1,1,1,1,1,1,1],
                 [0,0,0,0,0,0,0,0,0,0],
                 [1,1,1,1,1,1,1,1,1,1],
                 [0,0,0,0,0,0,0,0,0,0],
                 [1,1,1,1,1,1,1,1,1,1],
                 [0,0,0,0,0,0,0,0,0,0],
                 [1,1,1,1,1,1,1,1,1,1],
                 [0,0,0,0,0,0,0,0,0,0],
                 [1,1,1,1,1,1,1,1,1,1]])
"""
"""
#starter AEP511.25 1000iterations
matrix=np.array([[0,1,0,1,0,1,0,1,0,1],
                 [1,0,1,0,1,0,1,0,1,0],
                 [0,1,0,1,0,1,0,1,0,1],
                 [1,0,1,0,1,0,1,0,1,0],
                 [0,1,0,1,0,1,0,1,0,1],
                 [1,0,1,0,1,0,1,0,1,0],
                 [0,1,0,1,0,1,0,1,0,1],
                 [1,0,1,0,1,0,1,0,1,0],
                 [0,1,0,1,0,1,0,1,0,1],
                 [1,0,1,0,1,0,1,0,1,0]])
"""
"""
#  AEP507.12 5000it, 1000000 ITemp,2.5FTemp Name:Matrix-Gen1
matrix=np.array([[0,0,0,0,0,1,1,1,1,1],
                 [0,0,0,0,0,1,1,1,1,1],
                 [0,0,0,0,0,1,1,1,1,1],
                 [0,0,0,0,0,1,1,1,1,1],
                 [0,0,0,0,0,1,1,1,1,1],
                 [0,0,0,0,0,1,1,1,1,1],
                 [0,0,0,0,0,1,1,1,1,1],
                 [0,0,0,0,0,1,1,1,1,1],
                 [0,0,0,0,0,1,1,1,1,1],
                 [0,0,0,0,0,1,1,1,1,1]])
"""
"""
#GeneratedMatrix with AEP**10 IntT=100000 FinT=2.5 It=500 Gives AEP=513
matrixR=np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                 [1, 1, 0, 0, 1, 1, 0, 0, 0, 1],
                 [1, 0, 0, 1, 0, 0, 0, 1, 1, 0],
                 [1, 1, 0, 0, 0, 1, 1, 0, 0, 1],
                 [1, 0, 1, 0, 1, 0, 0, 0, 1, 0],
                 [1, 0, 0, 0, 0, 1, 0, 1, 0, 1],
                 [0, 1, 0, 1, 0, 0, 1, 0, 0, 1],
                 [1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
"""
"""
#Generated from Above matrix AEP=513.2 with AEP**10
matrix=np.array([[1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
                 [1, 0, 1, 0, 0, 0, 0, 0, 1, 0],
                 [0, 1, 0, 0, 1, 1, 0, 0, 0, 1],
                 [1, 0, 0, 1, 0, 0, 0, 1, 1, 0],
                 [0, 1, 0, 0, 0, 1, 1, 0, 0, 1],
                 [1, 0, 1, 0, 1, 0, 0, 0, 1, 0],
                 [1, 0, 1, 0, 0, 1, 0, 1, 0, 1],
                 [0, 1, 0, 1, 0, 0, 1, 0, 0, 1],
                 [1, 0, 1, 0, 0, 0, 0, 0, 1, 0],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
"""

manualPosition=pd.read_csv("ManualSolnA.csv")
matrixB=np.zeros((250,250))

for i in range(0,50):
    bi=int(manualPosition['x'][i]//16)
    bj=int(manualPosition['y'][i]//16)
    matrixB[bj][bi]=1
    for v in range(bj-24,bj+25):
        for u in range(bi-24,bi+25):
            if(v<0 or v>249 or u<0 or u>249):
                continue
            dist=math.sqrt((((v-bj)*16)**2)+(((u-bi)*16)**2)) #distance between cell and turbine at center
            if(dist>=400):
                continue
            if(matrixB[v][u]!=1):
                matrixB[v][u]-=1
                        
print(matrixB)


#plotting the matrix
print("Showing initial matrix")
plt.matshow(matrixB);
plt.colorbar()
plt.show()

"Calling Anneal Functions"
opt = BinaryAnnealer(matrixB)
print("\n")
optAnneal=opt.anneal()
arrayOneZero=optAnneal[0]
print(optAnneal)

"Converting solutions to X,Y 50x2 2d array format"
optimalTurbCoords=np.zeros((50,2))
s=0
for j in range(0,250):
    for i in range(0,250):
        if(arrayOneZero[j][i]==1): #CONCENTRATE!!!
            optimalTurbCoords[s][0]=(i*16)+8
            optimalTurbCoords[s][1]=(j*16)+8
            s=s+1
            
            
            """
            if((j*80)+40<50):#pushing from 40m at boundary to 50m
                optimalTurbCoords[s][0]=50
                optimalTurbCoords[s][1]=(i*80)+40
                s=s+1
            elif((i*80)+40<50):
                optimalTurbCoords[s][0]=(j*80)+40
                optimalTurbCoords[s][1]=50
                s=s+1
            elif((j*80)+40>3950):
                optimalTurbCoords[s][0]=3950
                optimalTurbCoords[s][1]=(i*80)+40
                s=s+1
            elif((i*80)+40>3950):
                optimalTurbCoords[s][0]=(j*80)+40
                optimalTurbCoords[s][1]=3950
                s=s+1
            else:
                optimalTurbCoords[s][0]=(j*80)+40
                optimalTurbCoords[s][1]=(i*80)+40
                #print(TurbCoords[s])
                s=s+1
            """
        """
        print("Showing final matrix")
        plt.matshow(arrayOneZero);
        plt.colorbar()
        plt.show()
        """
"FINAL OPTIMZED AEP"
print("Optimized AEP IS")
print("\n")
print(getAEP(turb_rad, optimalTurbCoords, power_curve, wind_inst_freq, 
            n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t))

###############
###Checking and verfying if 180 deg rotation of optimized position has lesser AEP
optimalTurbCoordsRot=np.zeros((50,2))
sx=0
for i in range(249,-1,-1):
    for j in range(249,-1,-1):
        if(arrayOneZero[i][j]==1): #CONCENTRATE!!!
            optimalTurbCoordsRot[sx][0]=(i*16)+8
            optimalTurbCoordsRot[sx][1]=(j*16)+8
            #print(TurbCoords[s])
            sx=sx+1
            
            
            """
            if((j*80)+40<50):#pushing from 40m at boundary to 50m
                optimalTurbCoordsRot[sx][0]=50
                optimalTurbCoordsRot[sx][1]=(i*80)+40
                s=s+1
            elif((i*80)+40<50):
                optimalTurbCoordsRot[sx][0]=(j*80)+40
                optimalTurbCoordsRot[sx][1]=50
                s=s+1
            elif((j*80)+40>3950):
                optimalTurbCoordsRot[sx][0]=3950
                optimalTurbCoordsRot[sx][1]=(i*80)+40
                s=s+1
            elif((i*80)+40>3950):
                optimalTurbCoordsRot[sx][0]=(j*80)+40
                optimalTurbCoordsRot[sx][1]=3950
                s=s+1
            else:
                optimalTurbCoordsRot[sx][0]=(j*80)+40
                optimalTurbCoordsRot[sx][1]=(i*80)+40
                #print(TurbCoords[s])
                sx=sx+1
            """
"FINAL OPTIMZED AEP"
print("verify if 180deg rotated optimial position has less AEP than our unrotated Optimized position")
print("Optimized AEP FOR 180 deg rotation IS")
print("\n")
print(getAEP(turb_rad, optimalTurbCoordsRot, power_curve, wind_inst_freq, 
            n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t))
np.savetxt("RotatedSoln.csv", optimalTurbCoordsRot, delimiter=",",header="x,y")
###################

"Pushing points on from boundary at 200m to 50m"
"""
pushedTurbCoords=np.zeros((50,2))
sp=0
for i in range(0,10):
    for j in range(0,10):
        if(arrayOneZero[i][j]==1): #CONCENTRATE!!!
            if(i!=0 and j!=0 and i!=9 and j!=0):
                ########
                #code for pushing from center 37.5 units due to free space due to corner bound
                if(i<=4 and j<=4):
                    pushedTurbCoords[sp][0]=(j*400)+200-37.5
                    pushedTurbCoords[sp][1]=(i*400)+200-37.5
                elif(i>4 and j<=4):
                    pushedTurbCoords[sp][0]=(j*400)+200-37.5
                    pushedTurbCoords[sp][1]=(i*400)+200+37.5
                elif(i<=4 and j>4):
                    pushedTurbCoords[sp][0]=(j*400)+200+37.5
                    pushedTurbCoords[sp][1]=(i*400)-200-37.5
                else:
                    pushedTurbCoords[sp][0]=(j*400)+200+37.5
                    pushedTurbCoords[sp][1]=(i*400)+200+37.5
                #############
                pushedTurbCoords[sp][0]=(j*400)+200 #Change i and j order if solution not matching
                pushedTurbCoords[sp][1]=(i*400)+200 #Change i and j order if solution not matching
                #print(TurbCoords[s][1])
                sp=sp+1
            elif(i==0 and j!=0 and j!=9):
                pushedTurbCoords[sp][0]=(j*400)+200
                pushedTurbCoords[sp][1]=50
                sp=sp+1
            elif(j==0 and i!=0 and i!=9):
                pushedTurbCoords[sp][0]=50
                pushedTurbCoords[sp][1]=(i*400)+200
                sp=sp+1
            elif(i==9 and j!=0 and j!=9):
                pushedTurbCoords[sp][0]=(j*400)+200
                pushedTurbCoords[sp][1]=3950
                sp=sp+1
            elif(j==9 and i!=0 and i!=9):
                pushedTurbCoords[sp][0]=3950
                pushedTurbCoords[sp][1]=(i*400)+200
                sp=sp+1
            elif(i==0 and j==0):
                pushedTurbCoords[sp][0]=50
                pushedTurbCoords[sp][1]=50
                sp=sp+1
            elif(i==9 and j==0):
                pushedTurbCoords[sp][0]=50
                pushedTurbCoords[sp][1]=3950
                sp=sp+1
            elif(i==0 and j==9):
                pushedTurbCoords[sp][0]=3950
                pushedTurbCoords[sp][1]=50
                sp=sp+1
            else:
                pushedTurbCoords[sp][0]=3950
                pushedTurbCoords[sp][1]=3950
                sp=sp+1


"FINAL PUSHED and OPTIMZED AEP"
print(" PUSHED and Optimized AEP IS")
print("\n")
print(getAEP(turb_rad, pushedTurbCoords, power_curve, wind_inst_freq, 
            n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t))
"""
"plot final solution"

x = optimalTurbCoords[:,0]
y = optimalTurbCoords[:,1]
colors = (0,0,0)
area = np.pi*3

# Plot
plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.title('Scatter plot Turbine Layout')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

"plot rotated soln"


x = optimalTurbCoordsRot[:,0]
y = optimalTurbCoordsRot[:,1]
colors = (0,0,0)
area = np.pi*3

# Plot
plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.title('Scatter plot Turbine Layout')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#PLEASE ROTATE OUTPUT SOLUTION BY 180 deg and check AEP

print("optimalXY coord:")
print(optimalTurbCoords)

"Saving to folder"
np.savetxt("Soln.csv", optimalTurbCoords, delimiter=",",header="x,y")
