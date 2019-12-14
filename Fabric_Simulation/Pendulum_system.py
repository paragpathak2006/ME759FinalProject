import pychrono as chrono
import MyGeometry as Geo
from pychrono import irrlicht as chronoirr
import numpy as np


def defaultParams(n):
    Orientx = chrono.ChMatrix33D()
    Orientx.SetMatr([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    Orienty = chrono.ChMatrix33D()
    Orienty.SetMatr([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

    # defaultOrientationx = [1.0, 0.0, 0.0, 0.0]
    # defaultOrientationy = [0, 0.0, 0.0, 1.0]

    posx = []
    posy = []
    pos = []

    for j in range(n):
        for i in range(n):
            p = [i, j, 0]

            px = [i, j - 0.5, 0]
            py = [i - 0.5, j, 0]

            pos.append(p)
            posx.append(px)
            posy.append(py)

    gz = [0.0, 0.0, -9.81]
    rod_length = 0.4
    return Orientx, Orienty, pos, posx, posy, rod_length, gz


## 0. Set the path to the Chrono data folder
chrono.SetChronoDataPath('./data/')
nGrid = 9
Orientx, Orienty, pos, posx, posy, rod_length, gz = defaultParams(nGrid)

system = Geo.setSys(gz)
system2 = Geo.setSys(gz)

ground = Geo.setGround(system)

crankx = []
cranky = []

for j in range(nGrid):
    for i in range(nGrid):
        # rodx = ground
        # rody = ground
        print(i, j, i + j * nGrid)
        pCons = [i - .5, j - .5, 0.0]
        if i < nGrid - 1:
            rodx = Geo.setCrank(i + j * nGrid, posx[i + j * nGrid], Orientx, rod_length, system)
            Geo.addSpringDampnersToGround(rodx, ground, system)

        if j < nGrid - 1:
            rody = Geo.setCrank(i + j * nGrid, posy[i + j * nGrid], Orienty, rod_length, system)
            Geo.addSpringDampnersToGround(rody, ground, system)

        if i != 0:
            Geo.addSphericalJoint(rodx, crankx[i - 1 + j * nGrid], pCons, system, True)
        else:
            Geo.addSphericalJoint(rodx, ground, pCons, system, False)

        if j != 0:
            Geo.addSphericalJoint(rody, cranky[i + (j - 1) * nGrid], pCons, system, True)

        crankx.append(rodx)
        cranky.append(rody)

        # if i == nGrid - 1:
        #     continue
        # if j == nGrid - 1:
        #     continue
        #  Cross connect grid
        if i < nGrid - 1 and j < nGrid - 1:
            Geo.addSphericalJoint(crankx[i + j * nGrid], cranky[i + j * nGrid], pCons, system, True)

        if i == nGrid - 1:
            Geo.addSphericalJoint(cranky[i + j * nGrid], crankx[i - 1 + j * nGrid], pCons, system, True)

        if j == nGrid - 1:
            Geo.addSphericalJoint(crankx[i + j * nGrid], cranky[i + (j - 1) * nGrid], pCons, system, True)

        #       engine_ground_crank spherical_crank_rod

## 4. Write the system hierarchy to the console (default log output destination)
system.ShowHierarchy(chrono.GetLog())

## 5. Prepare visualization with Irrlicht
##    Note that Irrlicht uses left-handed frames with Y up.
application = Geo.setApp(system)
while application.GetDevice().run():
    ## Initialize the graphical scene.
    application.BeginScene()

    ## Render all visualization objects.
    application.DrawAll()

    # chronoirr.ChIrrTools.drawSpring(application.GetVideoDriver(), .1, crank1.GetPos(),
    #                                 crank2.GetPos(),
    #                                 chronoirr.SColor(255, 80, 100, 100), 100, 10, False)
    ## Draw an XZ grid at the global origin to add in visualization.
    chronoirr.ChIrrTools.drawGrid(
        application.GetVideoDriver(), 1, 1, 20, 20,
        chrono.ChCoordsysD(chrono.ChVectorD(0, 0, 0), chrono.Q_from_AngX(chrono.CH_C_PI_2)),
        chronoirr.SColor(255, 80, 100, 100), True)

    ## Advance simulation by one step.
    application.DoStep()

    ## Finalize the graphical scene.
    application.EndScene()
