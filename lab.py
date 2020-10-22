import numpy as np
import pandas as pd
from numpy import sin, cos, pi
from itertools import product

E = 69000e6  # Elastic Modulus
v = 0.33     # Poisson's Ratio
Y = 275e6    # Yield Strength

C = np.array([[1/E, -v/E, 0], [-v/E, 1/E, 0], [0, 0, 2*(1+v)/E]])    # Stiffness tensor

# Max and min values for each dimension
l = [0.69992, 0.70008]
b = [0.038325, 0.038375]
h = [0.025715, 0.025765]
a = [0.2]
g1x = [0.34992, 0.35008]
g2x = [0.37492, 0.37508]
g2z = [0.01265, 0.01285]
g3x = [0.34992, 0.35008]
g4x = [0.44992, 0.45008]

ros1 = [0, 45, 90]
ros2 = [-45, 0, 45]

# Finds strain reading for a given angle
def strain1Rotate(epsilon,th):
    '''Rotate a strain epsilon =[e_x,e_y,gamma_xy] by an angle th given in radians.'''
    T = [cos(th)**2, sin(th)**2, sin(th)*cos(th)]
    return np.dot(T, epsilon)


# 3 point beam setup with defined mechanics
class ThreePointBeam:

    def __init__(self, length, base, height, g1x, g2x, g2z, g3x, g4x, load):
        self.l = length    # length in m
        self.b = base      # base in m
        self.h = height    # height in m
        self.p = load      # load in N
        self.g1x = g1x
        self.g2x = g2x
        self.g2z = g2z
        self.g3x = g3x
        self.g4x = g4x

    # Calculate moment of inertia
    def areaMomentOfInertia(self):
        I = self.b * self.h ** 3 / 12
        return I

    # Max deflection for 3-point bend
    def Deflection(self):
        w = self.p * self.l ** 3 / (48 * E * self.areaMomentOfInertia())
        return w

    # Shear force for 3-point bending
    def shearForce(self, x):
        if x <= self.l / 2:
            Q = self.p / 2
        else:
            Q = -self.p / 2
        return Q

    # Moment M for 3-points
    def bendingMoment(self, x):
        if x <= self.l / 2:
            M = self.p * x / 2
        else:
            M = self.p * (self.l - x) / 2
        return M

    # Axial stress by moment
    def axialStress(self, x, z):
        sigma = self.bendingMoment(x) * z / self.areaMomentOfInertia()
        sigma *= 1e-6
        return sigma

    # Shear stress
    def ShearStress(self, x, z):
        S = (self.shearForce(x) / 2 * self.areaMomentOfInertia()) * ((self.h ** 2 / 4) - z ** 2)
        S *= 1e-6
        return S

    # Strain readings for each gauge
    def strainReadings(self, x, z, ros_angles):
        sig = self.axialStress(x, z) * 1e6
        tau = self.ShearStress(x, z) * 1e6
        epsilon = np.dot(C, np.array([sig, 0, tau]))
        epsilon_rosette = np.array([strain1Rotate(epsilon, th * pi / 180) for th in ros_angles])
        epsilon_rosette *= 1e6
        return epsilon_rosette


# 4 point beam setup with defined mechanics
class FourPointBeam:

    def __init__(self, length, base, height, a, g1x, g2x, g2z, g3x, g4x, load):
        self.l = length    # length in m
        self.b = base      # base in m
        self.h = height    # height in m
        self.p = load      # load in N
        self.a = a
        self.g1x = g1x
        self.g2x = g2x
        self.g2z = g2z
        self.g3x = g3x
        self.g4x = g4x

    # Calculate moment of inertia
    def areaMomentOfInertia(self):
        I = self.b * self.h ** 3 / 12
        return I

    # Calculate max deflection
    def Deflection(self):
        w = ((self.p * self.a)/(48 * E * self.areaMomentOfInertia())) * ((3 * self.l**2) - (4 * self.a**2))
        return w

    # Shear force for 4-point bending
    def shearForce(self, x):
        if x <= self.a:
            Q = self.p / 2
        elif x > (self.l - self.a):
            Q = -self.p / 2
        else:
            Q = 0
        return Q

    # Bending moment for 4-point bending
    def bendingMoment(self, x):
        if x <= self.a:
            M = (self.p * x) / 2
        elif x > (self.l - self.a):
            M = (self.p * (x - self.l)) / 2
        else:
            M = (self.p * self.a) / 2
        return M

    # Axial stress by moment
    def axialStress(self, x, z):
        sigma = self.bendingMoment(x) * z / self.areaMomentOfInertia()
        sigma *= 1e-6
        return sigma

    # Shear stress
    def ShearStress(self, x, z):
        S = (self.shearForce(x) / 2 * self.areaMomentOfInertia()) * ((self.h ** 2 / 4) - z ** 2)
        S *= 1e-6
        return S

    # Strain readings for each gauge
    def strainReadings(self, x, z, ros_angles):
        sig = self.axialStress(x, z) * 1e6
        tau = self.ShearStress(x, z) * 1e6
        epsilon = np.dot(C, np.array([sig, 0, tau]))
        epsilon_rosette = np.array([strain1Rotate(epsilon, th * pi / 180) for th in ros_angles])
        epsilon_rosette *= 1e6
        return epsilon_rosette


# Handle error propagation through min/max approach
def interval(min, max):
    avg = (min + max) / 2
    #avg = round(avg, 3)
    error = (max - min) / 2
    #error = round(error, 3)
    value = "{} +/- {}".format(avg, error)
    return value


# Returns readings for a given beam
def RunLab(beam):
    gauges = {'G1': [beam.g1x, -beam.h / 2, ros1],
              'G2': [beam.g2x, beam.g2z - (beam.h / 2), ros1],
              'G3': [beam.g3x, beam.h / 2, ros1],
              'G4': [beam.g4x, -beam.h / 2, ros2]}

    output = {'Max Deflection': beam.Deflection()}
    for gauge in gauges:
        x = gauges[gauge][0]
        z = gauges[gauge][1]
        ros = gauges[gauge][2]
        reading = list(beam.strainReadings(x, z, ros))
        shear = beam.shearForce(x)
        axial_stress = beam.axialStress(x, z)
        shear_stress = beam.ShearStress(x, z)
        moment = beam.bendingMoment(x)

        output[gauge] = {"Shear Force": shear, "Moment": moment, "Axial Stress": axial_stress,
                         "Shear Stress": shear_stress, "a": reading[0], "b": reading[1], "c": reading[2]}

    return output


# Collects data for all possible dimension combinations and given load
def GetData(combos, load, type=''):

    w_list = []
    G_list = [{} for _ in range(4)]

    i = 0

    for combo in combos:
        if type == '3-point':
            beam = ThreePointBeam(*combo, load)
        elif type == '4-point':
            beam = FourPointBeam(*combo, load)
        else:
            Exception('Define beam type')

        data = RunLab(beam)
        w = data['Max Deflection']
        G1 = data['G1']
        G2 = data['G2']
        G3 = data['G3']
        G4 = data['G4']
        w_list.append(w)
        G_list[0][i] = G1
        G_list[1][i] = G2
        G_list[2][i] = G3
        G_list[3][i] = G4
        i += 1

    w_max = max(w_list)
    w_min = min(w_list)
    w_result = interval(w_min, w_max)
    j = 1
    print("\n\nLoad of {}".format(load))
    print('\nMax Deflection:{}'.format(w_result))
    for gauge in G_list:
        df = pd.DataFrame.from_dict(gauge, orient='index', columns=["Shear Force", "Moment", "Axial Stress",
                                                                      "Shear Stress", "a", "b", "c"])
        print("\nG{} Measurements".format(j))
        for measure in df:
            top = df[measure].max()
            bot = df[measure].min()
            out = interval(bot, top)
            print("{}: {}".format(measure, out))
        j += 1


# Loads applied in experiment
loads = [400, 800, 1200, 1600]

# Create list of possible dimensions due to measurement error
three_combo = list(product(l, b, h, g1x, g2x, g2z, g3x, g4x))
four_combo = list(product(l, b, h, a, g1x, g2x, g2z, g3x, g4x))

# Run analysis for 3-point and 4-point beam
print('Three Point Analysis')
for load in loads:
    GetData(three_combo, load, type="3-point")

print('Four Point Analysis')
for load in loads:
    GetData(four_combo, load, type="4-point")

